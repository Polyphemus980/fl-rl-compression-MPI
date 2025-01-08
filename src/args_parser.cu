#include <cstdio>
#include <cstring>

#include "args_parser.cuh"

namespace ArgsParser
{
    Args parseArguments(int argc, char **argv)
    {
        if (argc != 5)
        {
            usage(argv[0]);
        }

        Operation op;
        if (strcmp(argv[1], "c") == 0)
        {
            op = Operation::Compression;
        }
        else if (strcmp(argv[1], "d") == 0)
        {
            op = Operation::Decompression;
        }
        else
        {
            usage(argv[0]);
        }

        Method m;
        if (strcmp(argv[2], "fl") == 0)
        {
            m = Method::FixedLength;
        }
        else if (strcmp(argv[2], "rl") == 0)
        {
            m = Method::RunLength;
        }
        else
        {
            usage(argv[0]);
        }

        return Args{
            .operation = op,
            .method = m,
            .inputFile = argv[3],
            .outputFile = argv[4]};
    }

    void usage(const char *s)
    {
        fprintf(stderr, "USAGE: %s operation method input_file output_file\n");
        fprintf(stderr, "operation - c (compress) or d (decompress)\n");
        fprintf(stderr, "method - fl (fixed-length) or rl (run-length)\n");
        std::exit(1);
    }
} // ArgsParser