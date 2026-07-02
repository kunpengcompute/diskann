# 版本说明书

## 版本配套说明

### 产品版本信息

<a name="table62675726"></a>
<table><tbody><tr id="row41561572"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.1.1"><p id="p11044137"><a name="p11044137"></a><a name="p11044137"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.1.1 "><p id="p48427257"><a name="p48427257"></a><a name="p48427257"></a>Kunpeng BoostKit</p>
</td>
</tr>
<tr id="row24726251"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.2.1"><p id="p56669300"><a name="p56669300"></a><a name="p56669300"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.2.1 "><p id="p16166112734513"><a name="p16166112734513"></a><a name="p16166112734513"></a><span id="text1726192733514"><a name="text1726192733514"></a><a name="text1726192733514"></a>26.1.RC1</span></p>
</td>
</tr>
<tr id="row5497143514612"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.3.1"><p id="p162251517551"><a name="p162251517551"></a><a name="p162251517551"></a>软件名称</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.3.1 "><p id="p51757141375"><a name="p51757141375"></a><a name="p51757141375"></a>DiskANN</p>
</td>
</tr>
<tr id="row615762416269"><th class="firstcol" valign="top" width="42.17%" id="mcps1.1.3.4.1"><p id="p12158152417260"><a name="p12158152417260"></a><a name="p12158152417260"></a>软件版本</p>
</th>
<td class="cellrowborder" valign="top" width="57.830000000000005%" headers="mcps1.1.3.4.1 "><p id="p51757141375"><a name="p51757141375"></a><a name="p51757141375"></a>v1.0.0</p>
</td>
</tr>
</tbody>
</table>

### 与操作系统、编译器和CPU配套说明

| 操作系统 | CPU类型 | 内存 | 编译器 |
| --------- | -------- | ------ | -------- |
| openEuler 24.03 LTS SP3 | 鲲鹏950 7592C处理器 | 24\*64GB | GCC 12.3.1 |
| Debian 12 | 鲲鹏950 7592C处理器 | 24\*64GB | LLVM 16.0.6 |

## V1.0.0

### 更新说明

**新增特性**

| 编号 | 说明 |
| ------ | ------ |
| 1 | 两个补丁（等价/非等价）不可同时应用，需二选一。 |
| 2 | ARM64 NEON优化仅在AArch64架构下生效，x86\_64可编译运行但无法启用ARM优化。 |

**修改特性**

无

**删除特性**

无

### 已解决的问题

无

### 遗留问题

无

## 版本配套文档

### v1.0.0版本配套文档

| 文档名称 | 内容简介 | 交付形式 |
| --- | --- | --- |
| 《版本说明书》 | 介绍等价索引优化和非等价索引优化补丁的版本信息。 | 开源仓 |
| 《安装指南》 | 提供DiskANN的编译方法。 | 开源仓 |
| 《特性指南》 | 详细说明等价索引优化和非等价索引优化的技术内容。 | 开源仓 |
| 《最佳实践》 | 提供DiskANN基本使用指导与优化效果对比。 | 开源仓 |
| 《API参考》 | 对比原始DiskANN开源代码，详细列出接口变动。 | 开源仓 |

### 获取文档的方法

您可以通过[开源仓](https://gitcode.com/boostkit/diskann)浏览和获取相关文档。
