#include "include/sc_cli.h"
#include "include/sc_runner.h"

int main(int argc, char ** argv) {
    sc_args args;
    if (!sc_parse_args(argc, argv, args)) {
        return 1;
    }
    return sc_run(args);
}
