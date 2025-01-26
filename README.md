# Fixed-Length and Run-Length Compression

This repository contains implementations of Fixed-Length and Run-Length compression and decompression algorithms, developed using CUDA.

Detailed implementation information can be found in [IMPLEMENTATION-PLAN.md](./IMPLEMENTATION-PLAN.md) (note: the document is in Polish, as it was created as part of a university assignment).

## Building

To build this project, refer to the [CMakeLists.txt](./CMakeLists.txt) file.

## Running

Run the program with the following command:

```shell
./compress operation method input_file output_file
```

Parameters:

- `operation`: Specifies whether to compress (`c`) or decompress (`d`).
- `method`: Selects the compression method:
  - `fl`: Fixed-length compression using the GPU.
  - `fl-cpu`: Fixed-length compression using the CPU.
  - `rl`: Run-length compression using the GPU.
  - `rl-cpu`: Run-length compression using the CPU.
- `input_file`: Path to the input file.
- `output_file`: Path to the output file (contents will be overwritten if the file already exists).

## Testing

To run the tests, use the following command:

```shell
ctest
```
