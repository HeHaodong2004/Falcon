// Exact batch access to this workstation's installed Habitat/Bullet castRay.
// No geometry, intersection, visibility, or occlusion algorithm is replaced.
// The Python loader pins the complete Habitat ELF SHA-256 before entering here.
// This shim deliberately fails on unsupported binaries instead of guessing ABI.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {
// Mirrored POD ABI from Habitat v0.3.1 Geo.h / PhysicsManager.h.  Magnum's
// Vector3<float> is three contiguous floats, with no vtable or extra storage.
// The pinned binary, LP64/libstdc++ and static assertions bound this adapter.
struct NativeRay { float origin[3]; float direction[3]; };
struct NativeHit {
    int object_id;
    float point[3];
    float normal[3];
    double ray_distance;
};
struct NativeResults { std::vector<NativeHit> hits; NativeRay ray; };
static_assert(sizeof(void*) == 8, "Only the verified LP64 Habitat ABI is supported");
static_assert(sizeof(NativeRay) == 24, "Unexpected native Ray ABI");
static_assert(sizeof(NativeHit) == 40 && offsetof(NativeHit, ray_distance) == 32,
              "Unexpected native RayHitInfo ABI");
static_assert(sizeof(NativeResults) == 48, "Unexpected native RaycastResults ABI");
using CastFunction = NativeResults (*)(void*, const NativeRay&, double);
constexpr uintptr_t kCastRayOffset = 0x39f160;
// Signature of Simulator::castRay entry in the SHA-256-pinned installed ELF.
const unsigned char kCastRayPrefix[] = {0x53,0x48,0x8b,0x76,0x78,0x48,0x89,0xfb};

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using OffsetArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

struct Batch {
    FloatArray origin, directions;
    DoubleArray distances;
    size_t count;
    Batch(FloatArray o, FloatArray d, DoubleArray m)
      : origin(std::move(o)), directions(std::move(d)), distances(std::move(m)) {
        if (origin.ndim()!=1 || origin.shape(0)!=3 || directions.ndim()!=2 ||
            directions.shape(1)!=3 || distances.ndim()!=1 ||
            distances.shape(0)!=directions.shape(0))
            throw py::value_error("Expected origin[3], directions[N,3], max_distances[N]");
        count=static_cast<size_t>(directions.shape(0));
        for (size_t j=0;j<3;++j)
            if (!std::isfinite(origin.data()[j])) throw py::value_error("Nonfinite ray origin");
        for (size_t i=0;i<count;++i) {
            if (!std::isfinite(distances.data()[i]) || distances.data()[i]<0.)
                throw py::value_error("Ray max distances must be finite and nonnegative");
            for (size_t j=0;j<3;++j)
                if (!std::isfinite(directions.data()[3*i+j]))
                    throw py::value_error("Nonfinite ray direction");
        }
    }
    NativeRay ray(size_t i) const {
        NativeRay r;
        std::memcpy(r.origin,origin.data(),sizeof r.origin);
        std::memcpy(r.direction,directions.data()+3*i,sizeof r.direction);
        return r;
    }
};

class Engine {
    // Own the Python simulator throughout all native access, including calls.
    py::object owner_;
    void* simulator_ = nullptr;
    void* library_ = nullptr;
    CastFunction cast_ = nullptr;
    std::unordered_set<int> ignored(IntArray ids) const {
        if (ids.ndim()!=1) throw py::value_error("Expected ignored_ids[M]");
        return {ids.data(),ids.data()+ids.size()};
    }
    bool clear(const Batch& batch, size_t i, const std::unordered_set<int>& ignore) const {
        // Preserve Habitat's sorted ALL-hit results; a transparent ignored
        // robot/target-human hit never hides a later scene/human occluder.
        auto result=cast_(simulator_,batch.ray(i),batch.distances.data()[i]);
        const double cutoff=batch.distances.data()[i]-.015;
        for (const auto& hit:result.hits)
            if (!ignore.count(hit.object_id) && hit.ray_distance<cutoff) return false;
        return true;
    }
public:
    Engine(py::object simulator, const std::string& library_path) : owner_(std::move(simulator)) {
        auto bindings=py::module::import("habitat_sim._ext.habitat_sim_bindings");
        if (!py::isinstance(owner_,bindings.attr("Simulator")))
            throw py::type_error("Native batch requires a live native Habitat Simulator");
        auto* instance=reinterpret_cast<py::detail::instance*>(owner_.ptr());
        auto* type=py::detail::get_type_info(reinterpret_cast<PyTypeObject*>(bindings.attr("Simulator").ptr()));
        if (!type) throw std::runtime_error("Habitat pybind11 internals ABI is unavailable");
        auto value=instance->get_value_and_holder(type);
        if (!value || !value.type || !value.type->cpptype ||
            std::string(value.type->cpptype->name())!="N3esp3sim9SimulatorE" || !value.value_ptr())
            throw std::runtime_error(std::string("Unsupported Habitat pybind11 Simulator layout: ")+
                ((value && value.type && value.type->cpptype)?value.type->cpptype->name():"null"));
        simulator_=value.value_ptr();
        library_=dlopen(library_path.c_str(),RTLD_NOW|RTLD_NOLOAD);
        if (!library_) throw std::runtime_error("Habitat binding must already be loaded");
        void* init=dlsym(library_,"PyInit_habitat_sim_bindings");
        Dl_info info{};
        if (!init || !dladdr(init,&info) || !info.dli_fbase) {
            dlclose(library_);library_=nullptr;
            throw std::runtime_error("Cannot identify loaded Habitat ELF base");
        }
        const auto address=reinterpret_cast<uintptr_t>(info.dli_fbase)+kCastRayOffset;
        if (std::memcmp(reinterpret_cast<void*>(address),kCastRayPrefix,sizeof kCastRayPrefix)) {
            dlclose(library_);library_=nullptr;
            throw std::runtime_error("Pinned Habitat castRay entry signature differs");
        }
        cast_=reinterpret_cast<CastFunction>(address);
    }
    ~Engine() { if (library_) dlclose(library_); }

    py::array_t<bool> clear_rays(FloatArray origin, FloatArray directions,
                              DoubleArray distances, IntArray ids) const {
        Batch batch(std::move(origin),std::move(directions),std::move(distances));
        auto ignore=ignored(std::move(ids));
        py::array_t<bool> output(batch.count);
        for (size_t i=0;i<batch.count;++i) output.mutable_data()[i]=clear(batch,i,ignore);
        return output;
    }
    py::dict clear_groups(FloatArray origin, FloatArray directions,
                          DoubleArray distances, IntArray ids, OffsetArray offsets) const {
        Batch batch(std::move(origin),std::move(directions),std::move(distances));
        auto ignore=ignored(std::move(ids));
        if (offsets.ndim()!=1 || offsets.size()<1 || offsets.data()[0]!=0 ||
            offsets.data()[offsets.size()-1]!=static_cast<int64_t>(batch.count))
            throw py::value_error("Expected group_offsets starting at 0 and ending at ray count");
        for (py::ssize_t i=1;i<offsets.size();++i)
            if (offsets.data()[i]<offsets.data()[i-1])
                throw py::value_error("Group offsets must be nondecreasing");
        py::array_t<bool> output(offsets.size()-1);
        size_t ray_count=0;
        for (py::ssize_t group=0;group<offsets.size()-1;++group) {
            bool valid=true;
            for (int64_t i=offsets.data()[group];i<offsets.data()[group+1];++i) {
                ++ray_count;
                if (!clear(batch,i,ignore)) { valid=false;break; }
            }
            output.mutable_data()[group]=valid;
        }
        py::dict result;
        result["clear"]=std::move(output);result["rays"]=ray_count;
        return result;
    }
    py::dict nearest_hits(FloatArray origin, FloatArray directions,
                           DoubleArray distances, IntArray ids) const {
        Batch batch(std::move(origin),std::move(directions),std::move(distances));
        auto ignore=ignored(std::move(ids));
        py::array_t<bool> valid(batch.count);
        py::array_t<int> object_ids(batch.count);
        py::array_t<float> points({static_cast<py::ssize_t>(batch.count),py::ssize_t(3)});
        py::array_t<double> hit_distances(batch.count);
        for (size_t i=0;i<batch.count;++i) {
            valid.mutable_data()[i]=false;
            object_ids.mutable_data()[i]=-2;
            hit_distances.mutable_data()[i]=std::numeric_limits<double>::infinity();
            for (size_t j=0;j<3;++j) points.mutable_data()[3*i+j]=std::numeric_limits<float>::quiet_NaN();
            auto result=cast_(simulator_,batch.ray(i),batch.distances.data()[i]);
            // Existing native sort is authoritative, including tied distances.
            for (const auto& hit:result.hits) if (!ignore.count(hit.object_id)) {
                valid.mutable_data()[i]=true;
                object_ids.mutable_data()[i]=hit.object_id;
                hit_distances.mutable_data()[i]=hit.ray_distance;
                std::memcpy(points.mutable_data()+3*i,hit.point,sizeof hit.point);
                break;
            }
        }
        py::dict out;
        out["valid"]=std::move(valid);out["object_ids"]=std::move(object_ids);
        out["points"]=std::move(points);out["distances"]=std::move(hit_distances);
        return out;
    }
    py::dict cast_rays(FloatArray origin, FloatArray directions, DoubleArray distances) const {
        Batch batch(std::move(origin),std::move(directions),std::move(distances));
        std::vector<int64_t> offsets{0};std::vector<NativeHit> hits;
        for (size_t i=0;i<batch.count;++i) {
            auto result=cast_(simulator_,batch.ray(i),batch.distances.data()[i]);
            hits.insert(hits.end(),result.hits.begin(),result.hits.end());
            offsets.push_back(static_cast<int64_t>(hits.size()));
        }
        py::array_t<int64_t> output_offsets(offsets.size());
        py::array_t<int> objects(hits.size());
        py::array_t<double> output_distances(hits.size());
        py::array_t<float> points({static_cast<py::ssize_t>(hits.size()),py::ssize_t(3)});
        py::array_t<float> normals({static_cast<py::ssize_t>(hits.size()),py::ssize_t(3)});
        std::memcpy(output_offsets.mutable_data(),offsets.data(),offsets.size()*sizeof(int64_t));
        for (size_t i=0;i<hits.size();++i) {
            objects.mutable_data()[i]=hits[i].object_id;
            output_distances.mutable_data()[i]=hits[i].ray_distance;
            std::memcpy(points.mutable_data()+3*i,hits[i].point,sizeof hits[i].point);
            std::memcpy(normals.mutable_data()+3*i,hits[i].normal,sizeof hits[i].normal);
        }
        py::dict out;
        out["offsets"]=std::move(output_offsets);out["object_ids"]=std::move(objects);
        out["distances"]=std::move(output_distances);out["points"]=std::move(points);
        out["normals"]=std::move(normals);
        return out;
    }
};
} // namespace

PYBIND11_MODULE(_eventwm_native_batch,m) {
    m.doc()="Binary-pinned exact installed Habitat/Bullet batch raycasts";
    m.attr("abi_version")="habitat031_892f4d32_native_batch_v1";
    py::class_<Engine>(m,"Engine")
      .def(py::init<py::object,const std::string&>())
      .def("clear_rays",&Engine::clear_rays)
      .def("clear_groups",&Engine::clear_groups)
      .def("nearest_hits",&Engine::nearest_hits)
      .def("cast_rays",&Engine::cast_rays);
}
