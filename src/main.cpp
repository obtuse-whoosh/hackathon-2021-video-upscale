// waifu2x implemented with ncnn library

#include <stdio.h>
#include <algorithm>
#include <queue>
#include <vector>
#include <clocale>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// image decoder and encoder with stb
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_STDIO
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "webp_image.h"
#include <unistd.h> // getopt()

static std::vector<int> parse_optarg_int_array(const char* optarg)
{
    std::vector<int> array;
    array.push_back(atoi(optarg));

    const char* p = strchr(optarg, ',');
    while (p)
    {
        p++;
        array.push_back(atoi(p));
        p = strchr(p, ',');
    }

    return array;
}

// ncnn
#include "cpu.h"
#include "gpu.h"
#include "platform.h"

#include "waifu2x.h"

#include "filesystem_utils.h"

static void print_usage()
{
    fprintf(stdout, "Usage: waifu2x-ncnn-vulkan -i infile -o outfile [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -v                   verbose output\n");
    fprintf(stdout, "  -i input-path        input video path\n");
    fprintf(stdout, "  -o output-path       output video path\n");
    fprintf(stdout, "  -n noise-level       denoise level (-1/0/1/2/3, default=0)\n");
    fprintf(stdout, "  -s scale             upscale ratio (1/2/4/8/16/32, default=2)\n");
    fprintf(stdout, "  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu\n");
    fprintf(stdout, "  -m model-path        waifu2x model path (default=models-cunet)\n");
    fprintf(stdout, "  -g gpu-id            gpu device to use (-1=cpu, default=auto) can be 0,1,2 for multi-gpu\n");
    fprintf(stdout, "  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu\n");
    fprintf(stdout, "  -x                   enable tta mode\n");
    fprintf(stdout, "  -f                   FPS multiplier i.e frame interpolation (1=no interpolation, default=1)\n");
    // fprintf(stdout, "  -f format            output image format (jpg/png/webp, default=ext/png)\n");
}

class Task
{
public:
    int id;
    int scale;
    int frame;
    int duplicate_of;

    ncnn::Mat inimage;
    ncnn::Mat outimage;
};

class TaskQueue
{
public:
    TaskQueue()
    {
    }

    void put(const Task& v)
    {
        lock.lock();

        while (tasks.size() >= 8) // FIXME hardcode queue length
        {
            condition.wait(lock);
        }

        tasks.push(v);

        lock.unlock();

        condition.signal();
    }

    void get(Task& v)
    {
        lock.lock();

        while (tasks.size() == 0)
        {
            condition.wait(lock);
        }

        v = tasks.front();
        tasks.pop();

        lock.unlock();

        condition.signal();
    }

private:
    ncnn::Mutex lock;
    ncnn::ConditionVariable condition;
    std::queue<Task> tasks;
};

TaskQueue toproc;
TaskQueue tosave;

class LoadThreadParams
{
public:
    int scale;
    int jobs_load;
    cv::VideoCapture input_video;
    cv::VideoWriter output_video;
};

// Loads all video frames and queues up processing tasks
void* load(void* args)
{
    const LoadThreadParams* ltp = (const LoadThreadParams*)args;
    const int scale = ltp->scale;
    cv::VideoCapture reader = ltp->input_video;
    
    std::map<size_t, int>  frame_hashes;
    int frame_count = reader.get(cv::CAP_PROP_FRAME_COUNT);
    cv::Mat frame;
    #pragma omp parallel for schedule(static,1) num_threads(ltp->jobs_load)
    for (int i = 0; i < frame_count; i++)
    {
        // Read video frame
        if (!reader.read(frame))
        {
            continue;
        }

        // Define frame process task
        Task v;
        v.id = i;
        v.scale = scale;
        v.frame = i;

        // BGR => RGB
        int w = frame.cols;
        int h = frame.rows;
        int c = frame.channels();
        uint8_t* pixelPtr = (uint8_t*)frame.data;
        uint8_t* pixeldata = (uint8_t*)malloc(w * h * c);
        for(int i = 0; i < h; i++)
        {
            for(int j = 0; j < w; j++)
            {
                pixeldata[i*w*c + j*c + 2] /* R */ = pixelPtr[i*w*c + j*c + 0]; // B
                pixeldata[i*w*c + j*c + 1] /* G */ = pixelPtr[i*w*c + j*c + 1]; // G
                pixeldata[i*w*c + j*c + 0] /* B */ = pixelPtr[i*w*c + j*c + 2]; // R
            }
        }

        size_t key = std::_Hash_bytes(pixeldata, w * h * c, 0xdeadbeef);
        // TODO
        // if (frame_hashes.count(key))
        // {
        //     v.duplicate_of = frame_hashes[key];
        // } else 
        // {
        //     frame_hashes[key] = i;
            v.duplicate_of = -1;
        // }

        v.inimage = ncnn::Mat(w, h, (void*)pixeldata, (size_t)c, c);
        toproc.put(v);
    }

    return 0;
}

class ProcThreadParams
{
public:
    const Waifu2x* waifu2x;
};

// Proccesses video frames
void* proc(void* args)
{
    const ProcThreadParams* ptp = (const ProcThreadParams*)args;
    const Waifu2x* waifu2x = ptp->waifu2x;

    for (;;)
    {
        Task v;

        toproc.get(v);

        if (v.id == -233)
            break;

        if (v.duplicate_of >= 0)
        {
            tosave.put(v);
            continue;
        }

        const int scale = v.scale;
        if (scale == 1)
        {
            v.outimage = ncnn::Mat(v.inimage.w, v.inimage.h, (size_t)v.inimage.elemsize, (int)v.inimage.elemsize);
            waifu2x->process(v.inimage, v.outimage);

            tosave.put(v);
            continue;
        }

        int scale_run_count = 0;
        if (scale == 2)
        {
            scale_run_count = 1;
        }
        if (scale == 4)
        {
            scale_run_count = 2;
        }
        if (scale == 8)
        {
            scale_run_count = 3;
        }
        if (scale == 16)
        {
            scale_run_count = 4;
        }
        if (scale == 32)
        {
            scale_run_count = 5;
        }

        v.outimage = ncnn::Mat(v.inimage.w * 2, v.inimage.h * 2, (size_t)v.inimage.elemsize, (int)v.inimage.elemsize);
        waifu2x->process(v.inimage, v.outimage);

        for (int i = 1; i < scale_run_count; i++)
        {
            ncnn::Mat tmp = v.outimage;
            v.outimage = ncnn::Mat(tmp.w * 2, tmp.h * 2, (size_t)v.inimage.elemsize, (int)v.inimage.elemsize);
            waifu2x->process(tmp, v.outimage);
        }

        tosave.put(v);
    }

    return 0;
}

class SaveThreadParams
{
public:
    int verbose;
    int fps_mult;
    cv::VideoWriter output_video;
    std::string output_file;
};

class Frame
{
public:
    int num;
    int duplicate_of;
    cv::Mat frame;
};

bool compareFrame(Frame i1, Frame i2)
{
    return (i1.num < i2.num);
}

cv::Mat write(std::string output_file, cv::VideoWriter output_video, Frame frame, cv::Mat prev_frame, int fps_mult) {
  
  cv::Mat current_frame;
  if (frame.duplicate_of >= 0) 
  {
    cv::VideoCapture input_output_video(output_file);
    if (!input_output_video.isOpened())
    {
        fprintf(stderr, "\nFailed to open input_output_video%s\n", output_file.c_str());
    }
    
    input_output_video.set(cv::CAP_PROP_POS_FRAMES, frame.duplicate_of * fps_mult);
    input_output_video.read(current_frame);
    input_output_video.release();
  } else {
    current_frame = frame.frame;
  }

    if (frame.num == 233) { cv::imwrite("/home/jrobb/Downloads/lee_pre.png", prev_frame); } // TODO: remove
    for (int i = 1; frame.num > 0 && i < fps_mult; i++)
    {
        cv::Mat mid_frame;
        cv::addWeighted(prev_frame, 1 - (i / (double)fps_mult), current_frame , i / (double)fps_mult, 0, mid_frame);
        output_video.write(mid_frame);
        if (frame.num == 233) { cv::imwrite("/home/jrobb/Downloads/lee_"+std::to_string(i)+".png", mid_frame); } // TODO: remove
    }

    output_video.write(current_frame);
    if (frame.num == 233) { cv::imwrite("/home/jrobb/Downloads/lee_post.png", current_frame); } // TODO: remove
    return current_frame;
}

void* save(void* args)
{
    const SaveThreadParams* stp = (const SaveThreadParams*)args;
    const int verbose = stp->verbose;
    int fps_mult = stp->fps_mult;
    int dups = 0;
    int current_frame = 0;
    cv::Mat prev_frame;
    cv::VideoWriter output_video = stp->output_video;
    std::string output_file = stp->output_file;
    std::map<int, Frame> frames;

    fprintf(stdout, "Processing frame 0, duplicates 0\n");
    for (;;)
    {
        Task v;

        tosave.get(v);

        if (v.id == -233)
            break;

        // free input pixel data
        uint8_t * oldPixeldata = (uint8_t *)v.inimage.data;
        free(oldPixeldata);

        // RGB => BGR
        cv::Mat a = cv::Mat(v.outimage.h, v.outimage.w, CV_8UC3);
        unsigned char* pixelPtr = (unsigned char*)v.outimage.data;
        unsigned char* pixeldata = (unsigned char *)a.data;
        for(int i = 0; i < v.outimage.h; i++)
        {
            for(int j = 0; j < v.outimage.w; j++)
            {
                int offset = i*v.outimage.w*3 + j*3;
                pixeldata[offset + 2] /* R */ = pixelPtr[offset]; // B
                pixeldata[offset+ 1] /* G */ = pixelPtr[offset + 1]; // G
                pixeldata[offset] /* B */ = pixelPtr[offset + 2]; // R
            }
        }

        // add  frame
        Frame frame;
        frame.num = v.frame;
        frame.duplicate_of = v.duplicate_of;
        frame.frame = a;

        if (frame.duplicate_of >= 0)
        { 
            dups++; 
        }

        // process current frame
        if (frame.num == current_frame) {
            // if in order, write!
            current_frame++;
            prev_frame = write(output_file, output_video, frame, prev_frame, fps_mult);
            fprintf(stdout, "\033[A\33[2KT\rProcessing frame %d, duplicates %d, queued %lu\n", v.frame, dups, frames.size());
        } else {
            // queue it for precessing
            frames[frame.num] = frame;
        }

        // process queued up frames
        while (frames.count(current_frame)) {
            prev_frame = write(output_file, output_video, frame, prev_frame, fps_mult);
            fprintf(stdout, "\033[A\33[2KT\rProcessing frame %d, duplicates %d, queued %lu\n", v.frame, dups, frames.size());
            frames.erase(current_frame);
            current_frame++;
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    path_t inputpath;
    path_t outputpath;
    int noise = 0;
    int scale = 2;
    int fps_multiplier = 1;
    std::vector<int> tilesize;
    path_t model = PATHSTR("models-cunet");
    std::vector<int> gpuid;
    int jobs_load = 1;
    std::vector<int> jobs_proc;
    int jobs_save = 2;
    int verbose = 0;
    int tta_mode = 0;
    path_t format = PATHSTR("png");

    int opt;
    while ((opt = getopt(argc, argv, "i:o:n:s:t:m:g:j:f:vxh")) != -1)
    {
        switch (opt)
        {
        case 'i':
            inputpath = optarg;
            break;
        case 'o':
            outputpath = optarg;
            break;
        case 'n':
            noise = atoi(optarg);
            break;
        case 's':
            scale = atoi(optarg);
            break;
        case 't':
            tilesize = parse_optarg_int_array(optarg);
            break;
        case 'm':
            model = optarg;
            break;
        case 'g':
            gpuid = parse_optarg_int_array(optarg);
            break;
        case 'j':
            sscanf(optarg, "%d:%*[^:]:%d", &jobs_load, &jobs_save);
            jobs_proc = parse_optarg_int_array(strchr(optarg, ':') + 1);
            break;
        // case 'f':
        //     format = optarg;
        //     break;
        case 'v':
            verbose = 1;
            break;
        case 'x':
            tta_mode = 1;
            break;
        case 'f':
            fps_multiplier = atoi(optarg);
            break;
        case 'h':
        default:
            print_usage();
            return -1;
        }
    }

    if (inputpath.empty() || outputpath.empty())
    {
        print_usage();
        return -1;
    }

    if (noise < -1 || noise > 3)
    {
        fprintf(stderr, "invalid noise argument\n");
        return -1;
    }

    if (!(scale == 1 || scale == 2 || scale == 4 || scale == 8 || scale == 16 || scale == 32))
    {
        fprintf(stderr, "invalid scale argument\n");
        return -1;
    }

    if (tilesize.size() != (gpuid.empty() ? 1 : gpuid.size()) && !tilesize.empty())
    {
        fprintf(stderr, "invalid tilesize argument\n");
        return -1;
    }

    for (int i=0; i<(int)tilesize.size(); i++)
    {
        if (tilesize[i] != 0 && tilesize[i] < 32)
        {
            fprintf(stderr, "invalid tilesize argument\n");
            return -1;
        }
    }

    if (jobs_load < 1 || jobs_save < 1)
    {
        fprintf(stderr, "invalid thread count argument\n");
        return -1;
    }

    if (jobs_proc.size() != (gpuid.empty() ? 1 : gpuid.size()) && !jobs_proc.empty())
    {
        fprintf(stderr, "invalid jobs_proc thread count argument\n");
        return -1;
    }

    for (int i=0; i<(int)jobs_proc.size(); i++)
    {
        if (jobs_proc[i] < 1)
        {
            fprintf(stderr, "invalid jobs_proc thread count argument\n");
            return -1;
        }
    }

    if (path_is_directory(inputpath) || path_is_directory(outputpath))
    {
        fprintf(stderr, "Only single input is currently supported\n");
        return -1;
    }

    if (path_is_directory(outputpath))
    {
        fprintf(stderr, "Only single output is currently supported\n");
        return -1;
    }

    // collect input and output filepath
    path_t input_file = inputpath;
    path_t output_file = outputpath;
    
    int prepadding = 0;
    if (model.find(PATHSTR("models-cunet")) != path_t::npos)
    {
        if (noise == -1)
        {
            prepadding = 18;
        }
        else if (scale == 1)
        {
            prepadding = 28;
        }
        else if (scale == 2 || scale == 4 || scale == 8 || scale == 16 || scale == 32)
        {
            prepadding = 18;
        }
    }
    else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos)
    {
        prepadding = 7;
    }
    else if (model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
    {
        prepadding = 7;
    }
    else
    {
        fprintf(stderr, "unknown model dir type\n");
        return -1;
    }

    char parampath[256];
    char modelpath[256];
    if (noise == -1)
    {
        sprintf(parampath, "%s/scale2.0x_model.param", model.c_str());
        sprintf(modelpath, "%s/scale2.0x_model.bin", model.c_str());
    }
    else if (scale == 1)
    {
        sprintf(parampath, "%s/noise%d_model.param", model.c_str(), noise);
        sprintf(modelpath, "%s/noise%d_model.bin", model.c_str(), noise);
    }
    else if (scale == 2 || scale == 4 || scale == 8 || scale == 16 || scale == 32)
    {
        sprintf(parampath, "%s/noise%d_scale2.0x_model.param", model.c_str(), noise);
        sprintf(modelpath, "%s/noise%d_scale2.0x_model.bin", model.c_str(), noise);
    }

    path_t paramfullpath = sanitize_filepath(parampath);
    path_t modelfullpath = sanitize_filepath(modelpath);

    ncnn::create_gpu_instance();

    if (gpuid.empty())
    {
        gpuid.push_back(ncnn::get_default_gpu_index());
    }

    const int use_gpu_count = (int)gpuid.size();

    if (jobs_proc.empty())
    {
        jobs_proc.resize(use_gpu_count, 2);
    }

    if (tilesize.empty())
    {
        tilesize.resize(use_gpu_count, 0);
    }

    int cpu_count = std::max(1, ncnn::get_cpu_count());
    jobs_load = std::min(jobs_load, cpu_count);
    jobs_save = std::min(jobs_save, cpu_count);

    int gpu_count = ncnn::get_gpu_count();
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] < -1 || gpuid[i] >= gpu_count)
        {
            fprintf(stderr, "invalid gpu device\n");

            ncnn::destroy_gpu_instance();
            return -1;
        }
    }

    int total_jobs_proc = 0;
    for (int i=0; i<use_gpu_count; i++)
    {
        if (gpuid[i] == -1)
        {
            jobs_proc[i] = std::min(jobs_proc[i], cpu_count);
            total_jobs_proc += 1;
        }
        else
        {
            total_jobs_proc += jobs_proc[i];
        }
    }

    for (int i=0; i<use_gpu_count; i++)
    {
        if (tilesize[i] != 0)
            continue;

        if (gpuid[i] == -1)
        {
            // cpu only
            tilesize[i] = 4000;
            continue;
        }

        uint32_t heap_budget = ncnn::get_gpu_device(gpuid[i])->get_heap_budget();

        // more fine-grained tilesize policy here
        if (model.find(PATHSTR("models-cunet")) != path_t::npos)
        {
            if (heap_budget > 2600)
                tilesize[i] = 400;
            else if (heap_budget > 740)
                tilesize[i] = 200;
            else if (heap_budget > 250)
                tilesize[i] = 100;
            else
                tilesize[i] = 32;
        }
        else if (model.find(PATHSTR("models-upconv_7_anime_style_art_rgb")) != path_t::npos
            || model.find(PATHSTR("models-upconv_7_photo")) != path_t::npos)
        {
            if (heap_budget > 1900)
                tilesize[i] = 400;
            else if (heap_budget > 550)
                tilesize[i] = 200;
            else if (heap_budget > 190)
                tilesize[i] = 100;
            else
                tilesize[i] = 32;
        }
    }

    {
        std::vector<Waifu2x*> waifu2x(use_gpu_count);

        for (int i=0; i<use_gpu_count; i++)
        {
            int num_threads = gpuid[i] == -1 ? jobs_proc[i] : 1;

            waifu2x[i] = new Waifu2x(gpuid[i], tta_mode, num_threads);

            waifu2x[i]->load(paramfullpath, modelfullpath);

            waifu2x[i]->noise = noise;
            waifu2x[i]->scale = (scale >= 2) ? 2 : scale;
            waifu2x[i]->tilesize = tilesize[i];
            waifu2x[i]->prepadding = prepadding;
        }

        // main routine
        {
            // open input video
            cv::VideoCapture input_video(input_file);
            if (!input_video.isOpened())
            {
                fprintf(stderr, "Failed to open video %s\n", input_file.c_str());
                return -1;
            }

            // open output video
            cv::VideoWriter output_video;
            int codec = static_cast<int>(input_video.get(cv::CAP_PROP_FOURCC)); // codec
            int width = (int)input_video.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = (int)input_video.get(cv::CAP_PROP_FRAME_HEIGHT);
            cv::Size size(width * scale, height * scale); // size
            double fps = input_video.get(cv::CAP_PROP_FPS) * fps_multiplier; // fps
            output_video.open(output_file, codec, fps, size, true);
            fprintf(stdout, "%d x %d %ffps\n", width * scale, height * scale, fps);

            LoadThreadParams ltp;
            ltp.scale = scale;
            ltp.jobs_load = jobs_load;
            ltp.input_video = input_video;
            ltp.output_video = output_video;

            ncnn::Thread load_thread(load, (void*)&ltp);

            // waifu2x proc
            std::vector<ProcThreadParams> ptp(use_gpu_count);
            for (int i=0; i<use_gpu_count; i++)
            {
                ptp[i].waifu2x = waifu2x[i];
            }

            std::vector<ncnn::Thread*> proc_threads(total_jobs_proc);
            {
                int total_jobs_proc_id = 0;
                for (int i=0; i<use_gpu_count; i++)
                {
                    if (gpuid[i] == -1)
                    {
                        proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                    }
                    else
                    {
                        for (int j=0; j<jobs_proc[i]; j++)
                        {
                            proc_threads[total_jobs_proc_id++] = new ncnn::Thread(proc, (void*)&ptp[i]);
                        }
                    }
                }
            }

            // save image
            SaveThreadParams stp;
            stp.verbose = verbose;
            stp.output_video = output_video;
            stp.output_file = output_file;
            stp.fps_mult = fps_multiplier;

            std::vector<ncnn::Thread*> save_threads(jobs_save);
            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i] = new ncnn::Thread(save, (void*)&stp);
            }

            // end
            load_thread.join();

            Task end;
            end.id = -233;

            for (int i=0; i<total_jobs_proc; i++)
            {
                toproc.put(end);
            }

            for (int i=0; i<total_jobs_proc; i++)
            {
                proc_threads[i]->join();
                delete proc_threads[i];
            }

            for (int i=0; i<jobs_save; i++)
            {
                tosave.put(end);
            }

            for (int i=0; i<jobs_save; i++)
            {
                save_threads[i]->join();
                delete save_threads[i];
            }

            fprintf(stdout, "DONE?");
            input_video.release();
            output_video.release();
        }

        for (int i=0; i<use_gpu_count; i++)
        {
            delete waifu2x[i];
        }
        waifu2x.clear();
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
