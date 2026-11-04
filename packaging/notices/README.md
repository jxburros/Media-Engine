# Upstream notices omitted from crate archives

`ug-0.5.0/LICENSE-MIT` and `LICENSE-APACHE` are verbatim upstream notices
for `ug` and `ug-cuda` 0.5.0, from their published `.cargo_vcs_info.json` commit:

- https://github.com/LaurentMazare/ug/blob/8c6dd50d6e96a22db70e1462c0e49d0cda8294f7/LICENSE-MIT
- https://github.com/LaurentMazare/ug/blob/8c6dd50d6e96a22db70e1462c0e49d0cda8294f7/LICENSE-APACHE

The collector verifies the version, source repository, license declaration and
source commit before using these fallback files. Updating those crates requires
reviewing the notices again. The inventory follows normal binary dependencies;
build-only tools and development dependencies are not shipped in the archive.
In particular, `bindgen_cuda` is a build-only nvcc invocation helper, not linked
into the executable or embedded in its generated PTX.
