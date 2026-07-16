# Release Notes

## Version Mapping

### Product Information

<a name="table62675726"></a>
<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p48427257"><a name="p48427257"></a><a name="p48427257"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p16166112734513"><a name="p16166112734513"></a><a name="p16166112734513"></a><span id="text1726192733514"><a name="text1726192733514"></a><a name="text1726192733514"></a>26.1.RC1</span></p>
</td>
</tr>
<tr id="row5497143514612"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p162251517551"><a name="p162251517551"></a><a name="p162251517551"></a>Software Name</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p51757141375"><a name="p51757141375"></a><a name="p51757141375"></a>DiskANN</p>
</td>
</tr>
<tr id="row615762416269"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.4.1"><p id="p12158152417260"><a name="p12158152417260"></a><a name="p12158152417260"></a>Software Version</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.4.1 "><p id="p5175714137"><a name="p5175714137"></a><a name="p5175714137"></a>V1.0.0</p>
</td>
</tr>
</tbody>
</table>

### OS, Compiler, and CPU

| OS | CPU | Memory | Compiler |
| --------- | -------- | ------ | -------- |
| openEuler 24.03 LTS SP3 | Kunpeng 950 processor | 24 × 64 GB | GCC 12.3.1 |
| Debian 12 | Kunpeng 950 processor | 24 × 64 GB | LLVM 16.0.6 |

## V1.0.0

### Change Description

**New Features**

| No. | Description |
| ------ | ------ |
| 1 | Two patches (equivalence/non-equivalence) are mutually exclusive; you must choose one. |
| 2 | AArch64 NEON optimization is exclusive to the AArch64 architecture; while the code can still be compiled and run on x86_64, Arm-specific optimizations will be disabled. |

**Modified Features**

None

**Removed Features**

None

### Resolved Issues

None

### Known Issues

None

## Related Documentation

### v1.0.0 Related Documentation

| Document | Description | Delivery Method |
| --- | --- | --- |
| Release Notes | Provides version information for both equivalence and non-equivalence index optimization patches. | Open-source repository |
| Installation Guide | Provides instructions for compiling DiskANN. | Open-source repository |
| Feature Guide | Describes the technical details of equivalence index optimization and non-equivalence index optimization. | Open-source repository |
| Best Practices | Provides guidance on how to use DiskANN and showcases a benchmark comparison of optimization performance. | Open-source repository |
| API Reference | Lists the API changes compared with the original DiskANN open-source code. | Open-source repository |

### Obtaining Documentation

Visit the [open-source repository](https://gitcode.com/boostkit/diskann) to view or download related documents.
