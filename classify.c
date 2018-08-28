#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

void merge_image(image im1, image im2, image out)
{
    int size_im1 = im1.h * im1.w * im1.c * sizeof(float);
    int size_im2 = im2.h * im2.w * im2.c * sizeof(float);
    printf("%d %d\n",size_im1,size_im2);
    memcpy(out.data, im1.data, size_im1);
    memcpy((out.data + im1.h * im1.w * im1.c), im2.data, size_im2);
    printf("%d \n",out.w*out.h*out.c*4);
    for(int i=0;i<out.w*out.h*out.c;i++)
    {
        out.data[i] = out.data[i] * 2 - 1;
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }

        image im_front = load_image(input, net->w, net->h, 1);
        char im_bev_file[256];
        find_replace(input, "front_view", "birdeye_view", im_bev_file);
        image im_bev = load_image(im_bev_file, net->w, net->h, 1);
        image r = make_image(net->w, net->h, 2);
        merge_image(im_front, im_bev, r );
        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("result:%f, %f, %f, %f, %f\n",predictions[0],predictions[1],predictions[2],predictions[3],predictions[4]);
        
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        free_image(r);
        free_image(im_front);
        free_image(im_bev);
        if (filename) break;
    }
}

int main()
{
    char* cfgfile = "lidar.cfg";
    char *datacfg = "lidar.data";
    char *weightfile = "point-classify.weights";
    char* filename = "/home/adas/data/lidar-obj/png/car/front_view/id000_5315730.png";
    predict_classifier(datacfg, cfgfile, weightfile, filename, 1);
    return 0;
}