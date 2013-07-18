# -*- coding: utf-8 -*-

import re
import glob
import cv, cv2

class MemeClassifier:
    def __init__(self, meme_dname):
        self._meme_dname = meme_dname
        self._meme_fnames = glob.glob(meme_dname + '/*')
        self._meme_imgs = [cv.LoadImageM(fname) for fname in self._meme_fnames]

    def _calc_hist_inter(self, img1, img2):
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

    def _calc_sim(self, rgba1, rgba2):
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

        gray_hist_inter = self._calc_hist_inter(gray1, gray2)
        hue_hist_inter = self._calc_hist_inter(hue1, hue2)

        return (gray_hist_inter + hue_hist_inter) / 2.0

    def _fname_to_name(self, fname):
        fname = re.sub('^' + self._meme_dname + '\/', '', fname)
        fname = re.sub('\.jpg', '', fname)
        fname = re.sub('-', ' ', fname)
        return fname

    def classify(self, in_fname):
        in_img = cv.LoadImageM(in_fname)
        max_sim = 0.7
        max_idx = -1
        for idx, meme_img in enumerate(self._meme_imgs):
            battle = self._calc_sim(meme_img, in_img)
            if battle > max_sim:
                max_sim = battle
                max_idx = idx
        if max_idx == -1:
            return None
        return self._fname_to_name(self._meme_fnames[max_idx])

