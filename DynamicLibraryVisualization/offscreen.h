/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   offscreen.h
 * Author: eugene
 *
 * Created on October 11, 2017, 5:54 PM
 */

#ifndef OFFSCREEN_H
#define OFFSCREEN_H

#ifdef __cplusplus
extern "C"  
{
#endif
    enum OUTPUT_OPTION { PPM, PNG, MPEG };
    void set_output_windowsize(unsigned short window_size);
    void init_output(enum OUTPUT_OPTION output);
    void deinit_output(enum OUTPUT_OPTION output);
    void writeframe_output(enum OUTPUT_OPTION output);
#ifdef __cplusplus
}
#endif

#endif /* OFFSCREEN_H */
