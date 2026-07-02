# Introduction to DiskANN

## Latest Updates

- [2026-06-30]: The DiskANN optimization patches were released on the Gitcode platform, implementing both equivalence and non-equivalence index optimizations.

## Project Introduction

DiskANN is a disk-resident graph-based indexing algorithm proposed by the Microsoft research team for large-scale Approximate Nearest Neighbor (ANN) search. It enables high-performance, billion-scale vector retrieval on a single server with limited memory footprint. By caching Product Quantization (PQ) compressed vectors in memory while persisting the complete graph index and original full-precision vectors to Solid-State Drives (SSDs), DiskANN achieves exceptionally I/O-efficient search operations.
DiskANN was originally designed for the x86_64 architecture with AVX2 instruction sets. The Kunpeng optimization introduces intrusive modifications to the open-source DiskANN v0.7.0 codebase, successfully extending its capabilities to the AArch64 architecture. This porting delivers a comprehensive suite of hardware-centric performance enhancements, including NEON SIMD vectorization, data layout optimization (keeping adjacency lists in memory), asynchronous I/O pipelining, and refined ranking queue reduction.

## Directory Structure

The repository directory structure is as follows:

```text
diskann/
├─ docs                                   # Documentation directory
│   └── en                                # English document directory
│       ├── best_practices.md             # Best practices
│       ├── release_notes.md              # Release notes
│       ├── feature_introduction.md       # Feature introduction
│       ├── installation_guide.md         # Installation guide
│       └── api_reference.md              # API reference
├─ 0001-diskann_0.7.0-optimize-neq.patch  # Patch for non-equivalence index optimization (full-pipeline optimization)
├─ 0002-diskann_0.7.0-optimize-eqv.patch  # Patch for equivalence index optimization
└─ README_en.md                           # Project introduction file
```

## Version Description

For details about the DiskANN version updates, see [Release Notes](docs/en/release_notes.md).

## Documents

| Resource Name| Resource Description|
| ---- | ------- |
| [Release Notes](docs/en/release_notes.md)| Provides version information for both equivalence and non-equivalence index optimization patches.|
| [Feature Introduction](docs/en/feature_introduction.md)| Describes the technical details of equivalence index optimization and non-equivalence index optimization.|
| [Best Practices](docs/en/best_practices.md)| Provides guidance on how to use DiskANN and showcases a benchmark comparison of optimization performance.|
| [API Reference](docs/en/api_reference.md)| Lists the API changes compared with the original DiskANN open-source code.|
| [Installation Guide](docs/en/installation_guide.md)| Provides instructions for compiling DiskANN.|

## Disclaimer

This code repository contributes to the DiskANN open-source components. It strictly adheres to the coding style and methods, as well as security design of the native open-source software. Any vulnerability and security issues of the software shall be resolved by the corresponding upstream communities according to their response mechanisms. Please pay attention to the notifications and version updates released by the upstream communities. The Kunpeng computing community does not assume any responsibility for software vulnerabilities and security issues.

## License

- This project is licensed under the MIT license. For details, see [LICENSE](LICENSE).

- The documents of this project are licensed under CC-BY 4.0. For details, see [LICENSE](docs/en/LICENSE).

## Contribution Statement

We welcome your contributions to the community. If you have any questions/suggestions or want to provide feedback on feature requirements and bug reports, you can submit issues. For details, see Contribution Guideline. You are also welcome to share insights in [Discussions](https://gitcode.com/boostkit/community/discussions). Thank you for your support.

## Acknowledgments

DiskANN is jointly developed by the following Huawei department:

- Kunpeng Computing BoostKit Development Dept

Thank you to everyone in the community for your PRs. We warmly welcome contributions to DiskANN!
