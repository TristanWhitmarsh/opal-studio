import numpy as np
from skimage.measure import label, regionprops_table
from copy import deepcopy



class UBM:
    def __init__(self, carray):
        self._carray = carray

    def get_iou(self, m1, m2):

        # min over max
        xsi = np.shape(m1)[0]
        ysi = np.shape(m1)[1]
        mins = np.zeros_like(m1)
        maxs = np.zeros_like(m1)
        for ii in range(xsi):
            for jj in range(ysi):
                mins[ii, jj] = np.amin(np.array([m1[ii, jj], m2[ii, jj]]))
                maxs[ii, jj] = np.amax(np.array([m1[ii, jj], m2[ii, jj]]))

        return np.sum(mins.flatten()) / np.sum(maxs.flatten())

    def measure_area(self, im):

        label_img = label(im)
        props = regionprops_table(label_img, properties=("centroid", "area"))
        newd = {
            "x_loc": props["centroid-0"],
            "y_loc": props["centroid-1"],
            "area": props["area"],
        }

        return newd

    def find_stats(self, xx, endx, yy, endy):

        methno = np.shape(self._carray)[0]
        regions = []
        occupied = np.zeros((methno))
        astds = np.zeros((methno))
        means = np.zeros(methno)

        for mm in range(methno):
            seg = deepcopy(self._carray[mm, xx:endx, yy:endy])
            occupied[mm] = len(np.unique(seg.flatten()))
            dvals = self.measure_area(seg)
            areas = dvals["area"]
            means[mm] = np.median(areas)
            cdss = np.std(areas) / np.sqrt(len(areas))
            astds[mm] = 1.0 / (np.log(cdss) + 1.0)
            if np.isnan(astds[mm]):
                astds[mm] = 0
            # binary arrays for Jaccard score
            seg[seg > 0.0] = 1.0
            regions.append(seg)

        return regions, occupied, astds, means

    def find_winner_firstpass(self, regions):

        methno = np.shape(self._carray)[0]

        # matrix of jaccard scores
        jmat = np.zeros((methno, methno))
        for mm in range(methno):
            for nn in range(methno):
                jmat[mm, nn] = self.get_iou(regions[mm], regions[nn])
        sumj = np.nansum(jmat, axis=1)

        looser = np.where(sumj == np.amin(sumj))[0][0]
        # knock out the looser and then re-evaluate
        jmat[looser, :] = np.NaN
        jmat[:, looser] = np.NaN
        sumj = np.nansum(jmat, axis=1)

        # if only one method sees something then it defintely shouldn't
        # win just because it agrees with itself
        if np.amax(sumj) > 1.0:
            win = np.where(sumj == np.amax(sumj))[0][0]
        else:
            win = np.where(sumj == 0.0)[0][0]

        return win

    def find_clashes(self, xx, flatsa, flatsx, flatsy, flatsm, clashes, pairing, pairval, size_x, size_y):

        space = np.sqrt(flatsa[xx] / np.pi)
        diffsx = abs(flatsx - flatsx[xx])
        closex = np.where(diffsx < space)[0]
        diffsy = abs(flatsy - flatsy[xx])
        closey = np.where(diffsy < space)[0]
        borders = np.intersect1d(closex, closey)
        whoes = np.unique(flatsm[borders])
        # needs to intersect with more just itself
        if len(borders) > 1 and len(whoes) > 1:
            # just need to check that is actually an overlap
            xlow = int(np.amin(flatsx[borders]) - 2.0)
            if xlow < 0:
                xlow = 0
            ylow = int(np.amin(flatsy[borders]) - 2.0)
            if ylow < 0.0:
                ylow = 0
            xhi = int(np.amax(flatsx[borders]) + 2.0)
            if xhi > size_x:
                xhi = size_x
            yhi = int(np.amax(flatsy[borders]) + 2.0)
            if yhi > size_y:
                yhi = size_y
            values = np.zeros((xhi - xlow, yhi - ylow))
            for ii in range(len(borders)):
                thnu = int(flatsm[borders[ii]])
                values += self._carray[thnu, xlow:xhi, ylow:yhi]
            if (
                len(np.where(values.flatten() == 0.0)[0])
                < len(values.flatten()) - 3
            ):
                # anything more than 3 overlapping pixels is a clash
                clashes.extend(borders)
                pairing.extend(np.zeros((len(borders))) + pairval)
                pairval += 1

        return clashes, pairing, pairval

    def check_centres(self, labarray, win, xarr, yarr, areaarr, xx, yy, mini, nsize, savedm, savedx, savedy, saveda):

        winnerlist = np.where(labarray == win)[0]
        winxlow = np.where(xarr > xx - mini)[0]
        winxhi = np.where(xarr < xx + nsize + mini)[0]
        winylow = np.where(yarr > yy - mini)[0]
        winyhi = np.where(yarr < yy + nsize + mini)[0]

        comb1 = np.intersect1d(winnerlist, winxlow)
        comb2 = np.intersect1d(comb1, winxhi)
        comb3 = np.intersect1d(comb2, winylow)
        comb4 = np.intersect1d(comb3, winyhi)

        savedm.extend(labarray[comb4])
        savedx.extend(xarr[comb4])
        savedy.extend(yarr[comb4])
        saveda.extend(areaarr[comb4])

        return savedm, savedx, savedy, saveda

    def pick_star_end(self, xcens, ycens, nsize, size_x, size_y):

        starx = int(xcens[0] - nsize / 2)
        if starx < 0:
            starx = 0
        endx = int(xcens[0] + nsize / 2)
        if endx > size_x - nsize / 2:
            endx = size_x
        stary = int(ycens[0] - nsize / 2)
        if stary < 0:
            stary = 0
        endy = int(ycens[0] + nsize / 2)
        if endy > size_y - nsize / 2:
            endy = size_y

        return starx, stary, endx, endy

    def get_initial_ub(self, merit, methno, size_x, size_y, nsize):

        # get all contour catalogues
        xlist = []
        ylist = []
        arealist = []
        lablist = []
        for mm in range(methno):
            thisd = self.measure_area(self._carray[mm, :, :])
            lab = np.zeros((len(thisd["x_loc"]))) + mm
            xlist.extend(thisd["x_loc"])
            ylist.extend(thisd["y_loc"])
            arealist.extend(thisd["area"])
            lablist.extend(lab)

        # flatten, convert to arrays
        xarray = np.asarray(xlist)
        yarray = np.asarray(ylist)
        labarray = np.asarray(lablist)
        areaarray = np.asarray(arealist)

        savedx = []
        savedy = []
        savedm = []
        saveda = []

        for xx in range(0, size_x, nsize):
            for yy in range(0, size_y, nsize):

                if xx > size_x - nsize:
                    endx = size_x
                else:
                    endx = xx + nsize
                if yy > size_y - nsize:
                    endy = size_y
                else:
                    endy = yy + nsize

                regions, occupied, astds, means = self.find_stats(
                    xx, endx, yy, endy
                )

                if merit == "j1":
                    win = self.find_winner_firstpass(regions)

                elif merit == "pop":
                    win = np.where(occupied == np.amax(occupied))[0][0]

                elif merit == "cstd":
                    win = np.where(astds == np.amax(astds))[0][0]

                # if the winner has contour centres in this area
                # then those contours should go through.
                # "within" --> adding a buffer of a few pixels
                mini = np.ceil(np.percentile(areaarray, 32) * 0.05)  # grace in pixels
                savedm, savedx, savedy, saveda = self.check_centres(
                    labarray, win, xarray, yarray, areaarray, xx, yy, mini, nsize, savedm, savedx, savedy, saveda
                )

        flatsm = np.asarray(savedm)
        flatsx = np.asarray(savedx)
        flatsy = np.asarray(savedy)
        flatsa = np.asarray(saveda)

        return flatsm, flatsx, flatsy, flatsa

    def final_masks(
        self, joint_mask, method_mask, flatsx, flatsy, flatsm, cell_ID_ran, toodel
    ):

        xcen = flatsx.astype('int32')
        xcen[xcen == 0] = 1
        ycen = flatsy.astype('int32')
        ycen[ycen == 0] = 1

        labvals = [np.amax(
            self._carray[int(flatsm[ff]), xcen[ff] - 1: xcen[ff] + 1, ycen[ff] - 1: ycen[ff] + 1]
        ) for ff in range(len(flatsm))]

        cval = 0
        ch = 100
        flen = len(flatsm)
        for ff in range(flen):
            if labvals[ff] != 0 and ff not in toodel:

                if xcen[ff] - ch < 0:
                    initx = 0
                else:
                    initx = xcen[ff] - ch
                if ycen[ff] - ch < 0:
                    inity = 0
                else:
                    inity = ycen[ff] - ch

                if xcen[ff] + ch > np.shape(self._carray)[1]:
                    finix = np.shape(self._carray)[1]
                else:
                    finix = xcen[ff] + ch
                if ycen[ff] + ch > np.shape(self._carray)[2]:
                    finiy = np.shape(self._carray)[2]
                else:
                    finiy = ycen[ff] + ch

                vinds = self._carray[int(flatsm[ff]), initx:finix, inity:finiy] == labvals[ff]
                if len(vinds) > 10:
                    joint_mask[initx:finix, inity:finiy][vinds] = cell_ID_ran[cval]
                    method_mask[initx:finix, inity:finiy][vinds] = int(flatsm[ff]) + 1
                    cval += 1

        return joint_mask, method_mask, cval

    def form_um(self, merit="pop", nsize=80):
        """Form the ubermask from the combination of
        all other segmentation masks

        Parameters
        ----------
        carray : array
            stack of segmentation masks of size (n, x, y) where n
            is the number of masks and x and y denote image size
        merit : string
            criteria on which to perform combination i.e. 'pop' use
            mask which finds the largest number of cells, 'j1' use
            mask with the highest Jaccard score or 'cstd' use
            mask with smallest variance in cell areas.
        nsize : integer
            x and y dimensions of square region within the image
            across which to choose the optimum mask

        Returns
        -------
        joint_mask
            array containing ubermask
        method_mask
            array showing which method was selected to
            identify which cell. Methods are labelled as
            integers ascending from 1 and correspond to
            the order of methods listed in the config.yaml
        """

        methno, size_x, size_y = self._carray.shape
        joint_mask = np.zeros((size_x, size_y), dtype=int)
        method_mask = np.zeros((size_x, size_y)) + np.NaN

        flatsm, flatsx, flatsy, flatsa = self.get_initial_ub(
            merit, methno, size_x, size_y, nsize
        )
        print('first pass uber mask produced')
        print(len(flatsx))

        clashes = []
        pairing = []
        pairval = 1
        for xx in range(len(flatsx)):
            clashes, pairing, pairval = self.find_clashes(
                xx, flatsa, flatsx, flatsy, flatsm, clashes, pairing, pairval, size_x, size_y
            )

        print('identified clashes; fixing...')
        cell_ID_ran = np.arange(200, len(flatsm) + 200)
        np.random.shuffle(cell_ID_ran)
        cval = 0

        clasharray = np.asarray(clashes)
        print(f'identified {len(clasharray)} clashes and fixing if any')

        pairarray = np.asarray(pairing)
        arrs, unind = np.unique(clasharray, return_index=True)
        clasharray = clasharray[unind]
        pairarray = pairarray[unind]
        toodelete = []

        for ff in range(len(flatsm)):

            tooclose = np.where(clasharray == ff)[0]

            if len(tooclose) > 0:

                firstp = np.where(pairarray == pairarray[tooclose])[0]
                simfs = clasharray[firstp]
                xcens = flatsx[simfs]
                ycens = flatsy[simfs]

                starx, stary, endx, endy = self.pick_star_end(
                    xcens, ycens, nsize, size_x, size_y
                )

                regions, occupied, astds, means = self.find_stats(
                    starx, endx, stary, endy
                )

                if merit == "j1":
                    winner = self.find_winner_firstpass(regions)

                elif merit == "pop":
                    options = np.unique(flatsm[simfs])
                    ovals = occupied[options.astype("int")]
                    winner = options[np.where(ovals == np.amax(ovals))[0]]

                elif merit == "cstd":
                    options = np.unique(flatsm[simfs])
                    avals = astds[options.astype("int")]
                    winner = options[np.where(avals == np.amax(avals))[0]]

                if isinstance(winner, np.ndarray):
                    # if both are equally good just pick first:
                    winner = winner[0]

                # if the value stored in flatsm is not the winner it should not be plotted
                if flatsm[ff] != winner:
                    toodelete.append(ff)

        print('found winners')
        print(len(flatsx) - len(toodelete))
        joint_mask, method_mask, cval = self.final_masks(
            joint_mask, method_mask, flatsx, flatsy,
            flatsm, cell_ID_ran, toodelete
        )

        return joint_mask, method_mask
