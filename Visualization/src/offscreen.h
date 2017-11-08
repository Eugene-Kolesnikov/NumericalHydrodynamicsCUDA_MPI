/**
* @file offscreen.h
* @brief This header file contains the interface to the functions which save rendered
 * images to files: ppm, png, mpeg.
* @author Eugene Kolesnikov 
* @date 11/10/2017 
*/

#ifndef OFFSCREEN_H
#define OFFSCREEN_H

#ifdef __cplusplus
extern "C"  
{
#endif
    /**
    * @enum OUTPUT_OPTION
    * @brief Enum class which is responsible for the type of an output,
    */
    enum OUTPUT_OPTION { 
        /** 
         * @var PPM
         * @brief The type of output which will create a PPM image.
         * Better to use either for steady-state problems or for 
         * drawing images of systems at particular points of time.
         * Unless PNG, PPM has no compression, so it is very memory consuming.
        */
        PPM,
        
        /** 
         * @var PNG
         * @brief The type of output which will create a PNG image.
         * Better to use either for steady-state problems or for 
         * drawing images of systems at particular points of time.
        */
        PNG, 
        
        /** 
         * @var MPEG
         * @brief The type of output which will create an MPEG video.
         * Useful for representing properties which dynamically change in time.
        */
        MPEG 
    };
    
    /**
     * @brief Set's up the environment (window size) for the output image or video.
     * @param window_size -- width and height of the result image or video.
    */
    void set_output_windowsize(unsigned short window_size);
    
    /**
     * @brief Set's up the environment (output option).
     * @param output -- type of an output.
    */
    void init_output(enum OUTPUT_OPTION output, const char* path);
    
    /**
     * @brief Deinitializes the environment. (Necessary for MPEG option since
      * the encoder has to close the file appropriately).
     * @param output -- type of an output.
    */
    void deinit_output(enum OUTPUT_OPTION output);
    
    /**
     * @brief Either creates an image or writes a frame to a video file (depending
      * on the output option).
     * @param output -- type of an output.
    */
    void writeframe_output(enum OUTPUT_OPTION output, const char* path);
#ifdef __cplusplus
}
#endif

#endif /* OFFSCREEN_H */
