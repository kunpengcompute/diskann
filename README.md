# DiskANN介绍

## 最新消息

[2026.06.30]：DiskANN优化补丁发布于Gitcode平台，实现等价索引优化和非等价索引优化。

## 项目介绍

DiskANN是由微软团队提出的面向大规模向量近似最近邻搜索（ANN）的基于磁盘的图索引算法，能在有限内存下支持单机亿级向量检索，通过将PQ压缩向量驻留内存、完整索引与原始向量持久化于SSD实现IO高效的检索。
DiskANN实现是基于x86_64 AVX2指令集的，鲲鹏优化基于开源DiskANN v0.7.0代码做侵入式修改，将其扩展至ARM64（AArch64）架构，引入NEON SIMD向量化、数据布局优化（邻接表驻留内存）、异步IO流水化、精排队列缩减等多项性能优化。

## 目录结构

代码仓目录结构如下：

```text
diskann/
├─ docs                                   # 文档目录
│   └── zh                                # 中文文档目录
│       ├── best_practices.md             # 最佳实践
│       ├── release_notes.md              # 版本发布说明
│       ├── feature_introduction.md       # 特性介绍
│       ├── installation_guide.md         # 安装指南
│       └── api_reference.md              # API参考
├─ 0001-diskann_0.7.0-optimize-neq.patch  # 非等价索引优化补丁（全量优化）
├─ 0002-diskann_0.7.0-optimize-eqv.patch  # 等价索引优化补丁
└─ README.md                              # 项目介绍文件
```

## 版本说明

关于DiskANN的版本更新情况请参见《[版本说明书](docs/zh/release_notes.md)》。

## 学习文档

| 学习资源名称 | 学习资源简介 |
| ---- | ------- |
| [版本说明书](docs/zh/release_notes.md) | 介绍等价索引优化和非等价索引优化补丁的版本信息。 |
| [特性指南](docs/zh/feature_introduction.md) | 详细说明等价索引优化和非等价索引优化的技术内容。 |
| [最佳实践](docs/zh/best_practices.md) | 提供DiskANN基本使用指导与优化效果对比。 |
| [API参考](docs/zh/api_reference.md) | 对比原始DiskANN开源代码，详细列出接口变动。 |
| [安装指南](docs/zh/installation_guide.md) | 提供DiskANN的编译方法。 |

## 免责声明

此代码仓计划参与DiskANN开源组件，编码风格遵照原生开源软件，继承原生开源软件安全设计，不破坏原生开源软件设计及编码风格和方式，软件的任何漏洞与安全问题，均由相应的上游社区根据其漏洞和安全响应机制解决。请密切关注上游社区发布的通知和版本更新。鲲鹏计算社区对软件的漏洞及安全问题不承担任何责任。

## License

- 本项目采用MIT许可证授权，详见[LICENSE](LICENSE)文件。

- 本项目的文档适用CC-BY 4.0许可证，具体请参见文件[LICENSE](docs/zh/LICENSE)文件。

## 贡献声明

欢迎大家为社区做贡献，如果使用过程中有任何问题/建议，或者需要反馈特性需求和bug报告，可以提交[Issues](https://gitcode.com/boostkit/community/blob/master/docs/contributor/issue-submit.md)联系我们，具体贡献方法可参考[这里](https://gitcode.com/boostkit/community/blob/master/docs/contributor/contributing.md)。同时也欢迎大家在[讨论专区](https://gitcode.com/boostkit/community/discussions)展开讨论交流。感谢您的支持。

## 致谢

DiskANN由华为公司的下列部门联合贡献：

- 鲲鹏计算Boostkit开发部

感谢来自社区的每一个PR，欢迎贡献DiskANN！
