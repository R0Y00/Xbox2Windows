# pixel-binary check lists

Put one `<topic>.txt` file here for a check that needs LLVM 15. Each line is a script path plus
its arguments, written out in full (the inline `COMMON_LLVM` shortcut is not available here):

```
tests/my_check.py binary llvm-mc-15 llvm-objcopy-15 libLLVM-15.so.1
```

See [the check list rules](../README.md).
