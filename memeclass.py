# -*- coding: utf-8 -*-

import re
import glob
import cv

class MemeClassifier:
    def __init__(self, meme_dname):
        self._meme_dname = meme_dname
        self._meme_fnames = glob.glob(meme_dname + '/*')
        self._meme_imgs = [cv.LoadImageM(fname) for fname in self._meme_fnames]
        self._num_spatial_cut = 8
        self._all_sim_top_thresh = 0.4

        self._debug = False

    def _calc_one_hist_inter(self, hist1, hist2):
        inter = cv.CompareHist(hist1, hist2, cv.CV_COMP_INTERSECT)
        return inter

    def _calc_one_hist_head(self, hist):
        head = 0.0
        for i in range(8):
            head += hist.bins[i]
        return head

    def _calc_hist(self, img, max_val):
        assert img.type == cv.CV_8UC1
        hist = cv.CreateHist([128], cv.CV_HIST_ARRAY, [(0, max_val)], 1)
        cv.CalcHist([cv.GetImage(img)], hist)
        cv.NormalizeHist(hist, 1.0)
        return hist

    def _calc_all_hist_inter(self, rgba1, rgba2):        
        gray1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC1)
        gray2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC1)
        hsv1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC3)
        hsv2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC3)
        hue1 = cv.CreateMat(rgba1.rows, rgba1.cols, cv.CV_8UC1)
        hue2 = cv.CreateMat(rgba2.rows, rgba2.cols, cv.CV_8UC1)

        cv.CvtColor(rgba1, gray1, cv.CV_BGR2GRAY)
        cv.CvtColor(rgba2, gray2, cv.CV_BGR2GRAY)
        cv.CvtColor(rgba1, hsv1, cv.CV_BGR2HSV)
        cv.CvtColor(rgba2, hsv2, cv.CV_BGR2HSV)
        cv.Split(hsv1, hue1, None, None, None)
        cv.Split(hsv2, hue2, None, None, None)

        gray1_hist = self._calc_hist(gray1, 255)
        gray2_hist = self._calc_hist(gray2, 255)
        hue1_hist = self._calc_hist(hue1, 180)
        hue2_hist = self._calc_hist(hue2, 180)

        gray_hist_inter = self._calc_one_hist_inter(gray1_hist, gray2_hist)
        hue_hist_inter = self._calc_one_hist_inter(hue1_hist, hue2_hist)

        return (gray_hist_inter + hue_hist_inter) / 2.0

    def _calc_sim(self, splits1, splits2):
        total_sim = 0.0
        assert len(splits1) == len(splits2)
        len_splits = len(splits1)
        for row_idx in range(self._num_spatial_cut):
            for col_idx in range(self._num_spatial_cut):
                split_idx = row_idx * self._num_spatial_cut + col_idx
                split_sim = self._calc_all_hist_inter(splits1[split_idx], splits2[split_idx])
                if self._debug:
                    print split_sim
                total_sim += split_sim / (self._num_spatial_cut ** 2)
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
            if self._debug:
                print self._meme_fnames[idx]
            meme_splits = self._spatial_split(meme_img)
            battle = self._calc_sim(meme_splits, in_splits)
            if self._debug:
                print self._meme_fnames[idx], battle
            if battle > max_sim:
                max_sim = battle
                max_idx = idx
        if max_idx == -1:
            return None
        return self._fname_to_name(self._meme_fnames[max_idx])

