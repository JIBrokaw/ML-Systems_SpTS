#include <stdio.h>
#include <libufget.h>

int main(int argc, char *argv[])
{

    uf_collection_t *col = uf_collection_init();

    uf_collection_finalize(col);
    return 0;
}
