#include "pybind.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ai_engine_layer_norm", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
    // m.def("apx_layernorm", &layer_norm, "LayerNorm forward (CUDA)",
    //       py::arg("input"), py::arg("notmalized_shape"), py::arg("epsilon") = 1e-5);
    // m.def("ls_layer_norm_fp32",
    //     &ls_layer_norm, "Test kernel wrapper");
    m.def("flash_layernorm", &flash_layernorm, "Flash LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
     m.def("cutlass_layernorm", &cutlass_layernorm, "Cutlass LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
