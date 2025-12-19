#include "include/sd_cli.h"
#include "include/sd_runner.h"

int main(int argc, char ** argv) {
    sd_args args;
    if (!sd_parse_args(argc, argv, args)) {
        return 1;
    }
    return sd_run(args);
}
