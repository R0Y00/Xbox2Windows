Rust emitter reference outputs

ref_test.cpp builds fixed inputs in C++ and compares every output byte. The
fixture set was introduced in commit 7154f8c0b80682a153d38264a14b5833615c1b3b.
Both optimizer baselines in this directory were regenerated from upstream Rust
revision e9f435f9821ee8e03316367766c1551438aafec5 with the Xbox2Windows LLVM
producer identifier.

To reproduce in a temporary checkout of that revision:

1. Set SHADER_CRATE_MANIFEST to the absolute path of the Cargo.toml for the
   crate containing src/emit/mod.rs. Set REF_DIR to an absolute output directory.
   Copy ref_dump.rs beside mod.rs, add #[cfg(test)] mod ref_dump; there, and
   make build in src/emit/rect.rs visible as pub(super).
2. In src/emit/shader.rs, find the llvm.ident named metadata node. Set its
   producer to Xbox2Windows by changing the string argument in the preceding
   let ident = self.m.md_str(...) assignment. Keep the metadata node and the
   other DXIL version and shader-model fields intact.
3. Set SHADER_OPT_VAR to the environment variable name read by shader_opt()
   in src/emit/mod.rs. The default selects the normal optimization passes,
   and zero disables them. With the Rust dependencies available locally, run:

   mkdir -p "$REF_DIR/out" "$REF_DIR/out_opt0"
   (unset "$SHADER_OPT_VAR"; X2W_REF_OUT="$REF_DIR/out" cargo test --offline --manifest-path "$SHADER_CRATE_MANIFEST" --lib ref_dump)
   env "$SHADER_OPT_VAR=0" X2W_REF_OUT="$REF_DIR/out_opt0" cargo test --offline --manifest-path "$SHADER_CRATE_MANIFEST" --lib ref_dump

Each directory contains 12 binary files. Compare all 12 against the C++ emitter
using the matching optimizer setting.
