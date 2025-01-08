#ifndef ARGS_PARSER_H
#define ARGS_PARSER_H

namespace ArgsParser
{
    enum class Operation
    {
        Compression,
        Decompression,
    };

    enum class Method
    {
        RunLength,
        FixedLength,
    };

    struct Args
    {
        Operation operation;
        Method method;
        const char *inputFile;
        const char *outputFile;
    };

    Args parseArguments(int argc, char **argv);
    void usage(const char *s);
} // ArgsParser

#endif // ARGS_PARSER_H