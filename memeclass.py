# -*- coding: utf-8 -*-

import re
import glob
import cv, cv2

class MemeClassifier:
    def __init__(self, meme_dname):
        self._meme_dname = meme_dname
        self._meme_fnames = glob.glob(meme_dname + '/*')
        self._meme_imgs = [cv.LoadImageM(fname) for fname in self._meme_fnames]
        self._num_spatial_cut = 4
        self._spatial_sim_top_thresh = 0.4
        self._spatial_sim_bottom_thresh = 0.2
        self._all_sim_top_thresh = 0.55

    def _calc_one_hist_inter(self, img1, img2):
        assert img1.type == cv.CV_8UC1
        assert img2.type == cv.CV_8UC1
        hist1 = cv.CreateHist([32], cv.CV_HIST_ARRAY, [(0, 180)], 1)
        hist2 = cv.CreateHist([32], cv.CV_HIST_ARRAY, [(0, 180)], 1)
        cv.CalcHist([cv.GetImage(img1)], hist1)
        cv.CalcHist([cv.GetImage(img2)], hist2)
        cv.NormalizeHist(hist1, 1.0)
        cv.NormalizeHist(hist2, 1.0)
        inter = cv.CompareHist(hist1, hist2, cv.CV_COMP_INTERSECT)
        return inter

    def _calc_all_hist_inter(self, rgba1, rgba2):        
        gray1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC1)
        gray2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC1)
        hsv1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC3)
        hsv2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC3)
        hue1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC1)
        hue2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC1)
        sat1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC1)
        sat2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC1)

        cv.CvtColor(rgba1, gray1, cv.CV_BGR2GRAY)
        cv.CvtColor(rgba2, gray2, cv.CV_BGR2GRAY)
        cv.CvtColor(rgba1, hsv1, cv.CV_BGR2HSV)
        cv.CvtColor(rgba2, hsv2, cv.CV_BGR2HSV)
        cv.Split(hsv1, hue1, sat1, None, None)
        cv.Split(hsv2, hue2, sat2, None, None)

        gray_hist_inter = self._calc_one_hist_inter(gray1, gray2)
        hue_hist_inter = self._calc_one_hist_inter(hue1, hue2)
        sat_hist_inter = self._calc_one_hist_inter(sat1, sat2)

        return (gray_hist_inter + hue_hist_inter + sat_hist_inter) / 3.0

    def _calc_sim(self, splits1, splits2):
        total_sim = 0.0
        num_too_low = 0
        assert len(splits1) == len(splits2)
        len_splits = len(splits1)
        for row_idx in range(self._num_spatial_cut):
            for col_idx in range(self._num_spatial_cut):
                split_idx = row_idx * self._num_spatial_cut + col_idx
                split_sim = self._calc_all_hist_inter(splits1[split_idx], splits2[split_idx])
                #cv.ShowImage('in', splits2[split_idx])
                #cv.ShowImage('meme', splits1[split_idx])
                #cv.WaitKey(0)
                #print split_sim
                if split_sim > self._spatial_sim_top_thresh:
                    total_sim += split_sim / (self._num_spatial_cut ** 2)
                elif split_sim < self._spatial_sim_bottom_thresh:
                    num_too_low += 1
        if num_too_low > 4:
            return 0.0
        return total_sim

    def _spatial_split(self, img):
        res = []
        row_cut_len = img.rows / self._num_spatial_cut
        col_cut_len = img.cols / self._num_spatial_cut
        for row_start in range(self._num_spatial_cut):
            row_start *= row_cut_len
            for col_start in range(self._num_spatial_cut):
                col_start *= col_cut_len
                sub = cv.GetSubRect(img, (col_start, row_start, col_cut_len, row_cut_len))
                res.append(sub)
        return res

    def _is_long_post(self, img):
        return (float(img.rows) / img.cols > 2.0)

    def _fname_to_name(self, fname):
        fname = re.sub('^' + self._meme_dname + '\/', '', fname)
        fname = re.sub('\.jpg', '', fname)
        fname = re.sub('-', ' ', fname)
        return fname

    def classify(self, in_fname):
        in_img = cv.LoadImageM(in_fname)
        if self._is_long_post(in_img):
            return None
        in_splits = self._spatial_split(in_img)
        max_sim = self._all_sim_top_thresh
        max_idx = -1
        for idx, meme_img in enumerate(self._meme_imgs):
            #print self._meme_fnames[idx]
            meme_splits = self._spatial_split(meme_img)
            battle = self._calc_sim(meme_splits, in_splits)
            #print self._meme_fnames[idx], battle
            if battle > max_sim:
                max_sim = battle
                max_idx = idx
        if max_idx == -1:
            return None
        return self._fname_to_name(self._meme_fnames[max_idx])

