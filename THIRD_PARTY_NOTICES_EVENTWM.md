# EventWM publication licensing and provenance

The upstream Falcon/Habitat code retains its [MIT license](LICENSE) and original
copyright notices. Upstream project documentation is preserved in
[README_FALCON.md](README_FALCON.md). The publication copies the forecasting
implementation and its local adapter/data dependencies byte-for-byte from the
verified research source listed in `publication/source_identity.json`.

HM3D and Matterport3D are separate data dependencies, not MIT-licensed additions
to this repository. HM3D access is described by the [official HM3D repository](https://github.com/matterport/habitat-matterport-3dresearch),
which links the [Matterport academic-use agreement](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data).
That agreement restricts academic data and derived-information uses and requires
the agreement or its link with derived publications. Matterport3D acquisition and
its separate terms are documented by the [dataset authors](https://github.com/niessner/Matterport).
The [official Habitat dataset implementation](https://github.com/facebookresearch/habitat-matterport3d-dataset#license)
likewise distinguishes code licenses from trained-model/data terms.

This branch distributes implementation, academic aggregate findings, configuration
recipes and artifact identities. It does not redistribute scene meshes, images,
raw query/trajectory archives, target caches, NVIDIA runtime packages or trained
checkpoint binaries. We do not represent derived model weights as unrestricted
MIT software. The manifest identifies all six verified local checkpoints and
makes their external availability explicit. No public download is invented.

The synthetic demonstration history was authored for this publication and contains
no collected observations, real scene geometry, names, addresses or person data.
Published figures here show aggregate calibration and training curves. The local
research release also retains qualitative scene/trajectory diagnostics; those
larger derived artifacts are not bundled into Git.

Licensed raw assets and the frozen training archives remain necessary for full
scientific reproduction. Existing users must retain their original dataset
agreements and attribution. This inventory records what was checked and what was
excluded; it does not grant rights to third-party data or redistribute a new
license for them. License sources were inspected on 2026-10-15.
