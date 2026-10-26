// One pass over already validated legal graph cells; no simulator/map inputs.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>

struct Buffer {Py_buffer value={};bool open=false;~Buffer(){if(open)PyBuffer_Release(&value);}bool get(PyObject* o){open=PyObject_GetBuffer(o,&value,PyBUF_CONTIG)==0;return open;}};
static PyObject* extract(PyObject*,PyObject* args){
    PyObject *cells,*order,*first,*last,*ids;if(!PyArg_ParseTuple(args,"OOOOO",&cells,&order,&first,&last,&ids))return nullptr;
    if(!PyList_CheckExact(cells))Py_RETURN_NONE;
    Py_ssize_t n=PyList_Size(cells);Buffer ob,fb,lb,ib;
    if(!ob.get(order)||!fb.get(first)||!lb.get(last)||!ib.get(ids)){PyErr_Clear();Py_RETURN_NONE;}
    if(ob.value.len!=n*8||fb.value.len!=n*8||lb.value.len!=n*8||ib.value.len!=n*4)Py_RETURN_NONE;
    auto ord=static_cast<const int64_t*>(ob.value.buf);auto f=static_cast<double*>(fb.value.buf);auto l=static_cast<double*>(lb.value.buf);auto sid=static_cast<int32_t*>(ib.value.buf);
    PyObject* names=PyList_New(0);if(!names)return nullptr;std::unordered_map<std::string,int> lookup;
    for(Py_ssize_t i=0;i<n;i++){
        if(ord[i]<0||ord[i]>=n){Py_DECREF(names);Py_RETURN_NONE;}
        PyObject* cell=PyList_GET_ITEM(cells,ord[i]);
        if(!PyDict_CheckExact(cell)||PyDict_Size(cell)!=6){Py_DECREF(names);Py_RETURN_NONE;}
        PyObject *source=PyDict_GetItemString(cell,"source"),*a=PyDict_GetItemString(cell,"first_seen"),*b=PyDict_GetItemString(cell,"last_seen");
        if(!source||!PyUnicode_CheckExact(source)||!a||!b){Py_DECREF(names);Py_RETURN_NONE;}
        if(!PyDict_GetItemString(cell,"cell")||!PyDict_GetItemString(cell,"state")||!PyDict_GetItemString(cell,"height")){Py_DECREF(names);Py_RETURN_NONE;}
        if((!PyFloat_CheckExact(a)&&!PyLong_CheckExact(a))||(!PyFloat_CheckExact(b)&&!PyLong_CheckExact(b))){Py_DECREF(names);Py_RETURN_NONE;}
        Py_ssize_t length;const char* text=PyUnicode_AsUTF8AndSize(source,&length);f[i]=PyFloat_AsDouble(a);l[i]=PyFloat_AsDouble(b);
        if(!text||PyErr_Occurred()||!std::isfinite(f[i])||!std::isfinite(l[i])){PyErr_Clear();Py_DECREF(names);Py_RETURN_NONE;}
        std::string key(text,length);auto found=lookup.find(key);
        if(found==lookup.end()){int id=lookup.size();lookup.emplace(key,id);if(PyList_Append(names,source)){Py_DECREF(names);return nullptr;}sid[i]=id;}else sid[i]=found->second;
    }
    return names;
}
static PyObject* columns(PyObject*,PyObject* args){
    PyObject *cells,*keys,*states,*heights;if(!PyArg_ParseTuple(args,"OOOO",&cells,&keys,&states,&heights))return nullptr;
    if(!PyList_CheckExact(cells))Py_RETURN_NONE;
    Py_ssize_t n=PyList_Size(cells);Buffer kb,sb,hb;
    if(!kb.get(keys)||!sb.get(states)||!hb.get(heights)){PyErr_Clear();Py_RETURN_NONE;}
    if(kb.value.len!=n*16||sb.value.len!=n||hb.value.len!=n*8)Py_RETURN_NONE;
    auto k=static_cast<int64_t*>(kb.value.buf);auto s=static_cast<uint8_t*>(sb.value.buf);auto h=static_cast<double*>(hb.value.buf);
    for(Py_ssize_t i=0;i<n;i++){
        PyObject* cell=PyList_GET_ITEM(cells,i);if(!PyDict_CheckExact(cell))Py_RETURN_NONE;
        PyObject *key=PyDict_GetItemString(cell,"cell"),*state=PyDict_GetItemString(cell,"state"),*height=PyDict_GetItemString(cell,"height");
        if(!key||!state||!height||!PyLong_CheckExact(state)||(!PyFloat_CheckExact(height)&&!PyLong_CheckExact(height)))Py_RETURN_NONE;
        bool tuple=PyTuple_CheckExact(key);if(!tuple&&!PyList_CheckExact(key))Py_RETURN_NONE;
        if((tuple?PyTuple_GET_SIZE(key):PyList_GET_SIZE(key))!=2)Py_RETURN_NONE;
        for(int a=0;a<2;a++){
            PyObject* value=tuple?PyTuple_GET_ITEM(key,a):PyList_GET_ITEM(key,a);if(!PyLong_CheckExact(value))Py_RETURN_NONE;
            long long v=PyLong_AsLongLong(value);if(PyErr_Occurred()){PyErr_Clear();Py_RETURN_NONE;}
            if(v<INT32_MIN||v>INT32_MAX)Py_RETURN_NONE;k[2*i+a]=v;
        }
        long label=PyLong_AsLong(state);if(PyErr_Occurred()){PyErr_Clear();Py_RETURN_NONE;}if(label!=1&&label!=2)Py_RETURN_NONE;
        s[i]=label;h[i]=PyFloat_AsDouble(height);if(PyErr_Occurred()||!std::isfinite(h[i])){PyErr_Clear();Py_RETURN_NONE;}
    }
    Py_RETURN_TRUE;
}
static PyMethodDef methods[]={{"extract",extract,METH_VARARGS,"Validated canonical metadata columns, or None for the original fallback."},{"columns",columns,METH_VARARGS,"Canonical integer cell/state/height columns, or None for the original fallback."},{nullptr,nullptr,0,nullptr}};
static PyModuleDef definition={PyModuleDef_HEAD_INIT,"_falcon_native_metadata",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__falcon_native_metadata(){return PyModule_Create(&definition);}
