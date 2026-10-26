// Optional OptiX7.7 bridge. All CUDA buffers/streams are owned by Torch.
// Only measured-map AABBs are supplied; no scene assets or simulator access.
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <cstdio>
#include <memory>

struct State {
    OptixDeviceContext context=nullptr;
    OptixModule module=nullptr;
    OptixProgramGroup groups[3]={};
    OptixPipeline pipeline=nullptr;
    ~State(){
        if(pipeline)optixPipelineDestroy(pipeline);
        for(auto p:groups)if(p)optixProgramGroupDestroy(p);
        if(module)optixModuleDestroy(module);
        if(context)optixDeviceContextDestroy(context);
    }
};
#define CHECK(call) do {OptixResult r=(call);if(r){snprintf(error,error_size,"%s:%d %s",#call,int(r),optixGetErrorName(r));return int(r);}}while(0)

extern "C" int falcon_optix_create(const char* ptx,size_t bytes,void** returned,char* error,size_t error_size){
    CHECK(optixInit());
    CUcontext cuda=nullptr;auto cr=cuCtxGetCurrent(&cuda);
    if(cr||!cuda){snprintf(error,error_size,"Torch CUDA context required:%d",cr);return -1;}
    std::unique_ptr<State>s(new State);OptixDeviceContextOptions options={};
    CHECK(optixDeviceContextCreate(cuda,&options,&s->context));
    OptixModuleCompileOptions mc={};mc.optLevel=OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;mc.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    OptixPipelineCompileOptions pc={};pc.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;pc.numPayloadValues=1;pc.numAttributeValues=0;pc.pipelineLaunchParamsVariableName="params";pc.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    size_t log_size=error_size;auto r=optixModuleCreate(s->context,&mc,&pc,ptx,bytes,error,&log_size,&s->module);
    if(r)return int(r);
    OptixProgramGroupDesc descriptions[3]={};
    descriptions[0].kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;descriptions[0].raygen.module=s->module;descriptions[0].raygen.entryFunctionName="__raygen__main";
    descriptions[1].kind=OPTIX_PROGRAM_GROUP_KIND_MISS;descriptions[1].miss.module=s->module;descriptions[1].miss.entryFunctionName="__miss__main";
    descriptions[2].kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;descriptions[2].hitgroup.moduleCH=s->module;descriptions[2].hitgroup.entryFunctionNameCH="__closesthit__main";descriptions[2].hitgroup.moduleIS=s->module;descriptions[2].hitgroup.entryFunctionNameIS="__intersection__leaf";
    OptixProgramGroupOptions pg={};log_size=error_size;
    r=optixProgramGroupCreate(s->context,descriptions,3,&pg,error,&log_size,s->groups);if(r)return int(r);
    OptixPipelineLinkOptions link={};link.maxTraceDepth=1;log_size=error_size;
    r=optixPipelineCreate(s->context,&pc,&link,s->groups,3,error,&log_size,&s->pipeline);if(r)return int(r);
    OptixStackSizes stack={};for(auto p:s->groups)CHECK(optixUtilAccumulateStackSizes(p,&stack,s->pipeline));
    unsigned from_traversal=0,from_state=0,continuation=0;CHECK(optixUtilComputeStackSizes(&stack,1,0,0,&from_traversal,&from_state,&continuation));
    CHECK(optixPipelineSetStackSize(s->pipeline,from_traversal,from_state,continuation,1));
    *returned=s.release();return 0;
}

static OptixBuildInput input(CUdeviceptr* boxes,unsigned count,unsigned* flags){
    OptixBuildInput in={};in.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;in.customPrimitiveArray.aabbBuffers=boxes;in.customPrimitiveArray.numPrimitives=count;in.customPrimitiveArray.strideInBytes=sizeof(OptixAabb);in.customPrimitiveArray.flags=flags;in.customPrimitiveArray.numSbtRecords=1;return in;
}
static OptixAccelBuildOptions build_options(){OptixAccelBuildOptions options={};options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;options.operation=OPTIX_BUILD_OPERATION_BUILD;return options;}

extern "C" int falcon_optix_sizes(void* pointer,unsigned long long box_pointer,unsigned count,unsigned long long* output,unsigned long long* temporary,char* error,size_t error_size){
    auto s=static_cast<State*>(pointer);CUdeviceptr boxes=box_pointer;unsigned flags=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;auto in=input(&boxes,count,&flags);auto options=build_options();OptixAccelBufferSizes sizes={};CHECK(optixAccelComputeMemoryUsage(s->context,&options,&in,1,&sizes));*output=sizes.outputSizeInBytes;*temporary=sizes.tempSizeInBytes;return 0;
}
extern "C" int falcon_optix_build(void* pointer,unsigned long long box_pointer,unsigned count,unsigned long long temporary,size_t temporary_bytes,unsigned long long output,size_t output_bytes,unsigned long long stream,unsigned long long* handle,char* error,size_t error_size){
    auto s=static_cast<State*>(pointer);CUdeviceptr boxes=box_pointer;unsigned flags=OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;auto in=input(&boxes,count,&flags);auto options=build_options();CHECK(optixAccelBuild(s->context,reinterpret_cast<CUstream>(stream),&options,&in,1,temporary,temporary_bytes,output,output_bytes,handle,nullptr,0));return 0;
}
extern "C" int falcon_optix_headers(void* pointer,void* headers,char* error,size_t error_size){
    auto s=static_cast<State*>(pointer);for(int i=0;i<3;i++)CHECK(optixSbtRecordPackHeader(s->groups[i],static_cast<char*>(headers)+i*OPTIX_SBT_RECORD_HEADER_SIZE));return 0;
}
extern "C" int falcon_optix_launch(void* pointer,unsigned long long records,unsigned long long parameters,size_t parameter_bytes,unsigned width,unsigned long long stream,char* error,size_t error_size){
    auto s=static_cast<State*>(pointer);OptixShaderBindingTable sbt={};sbt.raygenRecord=records;sbt.missRecordBase=records+OPTIX_SBT_RECORD_HEADER_SIZE;sbt.missRecordStrideInBytes=OPTIX_SBT_RECORD_HEADER_SIZE;sbt.missRecordCount=1;sbt.hitgroupRecordBase=records+2*OPTIX_SBT_RECORD_HEADER_SIZE;sbt.hitgroupRecordStrideInBytes=OPTIX_SBT_RECORD_HEADER_SIZE;sbt.hitgroupRecordCount=1;
    CHECK(optixLaunch(s->pipeline,reinterpret_cast<CUstream>(stream),parameters,parameter_bytes,&sbt,width,1,1));return 0;
}
extern "C" void falcon_optix_destroy(void* pointer){delete static_cast<State*>(pointer);}
