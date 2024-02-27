#  Holmanu Antonio-Marius
#  Grupa 462, CTI

import numpy as np
import cv2 as cv
import os


def show_image(title, image):
    # se comenteaza pentru afisarea imaginilor mici si decomenteaza pentru afisarea imaginilor mari
    image = cv.resize(image, (0, 0), fx=0.18, fy=0.18)
    cv.imshow(title, image)
    cv.moveWindow(title, 200, 100)
    cv.waitKey(0)
    cv.destroyAllWindows()


#  am definit cateva filtre pentru convertirea la HSV
lower_white_board = (0, 140, 0)
upper_white_board = (255, 255, 70)

lower_white_filter1 = (0, 0, 230)
upper_white_filter1 = (255, 100, 255)

lower_white_filter2 = (70, 0, 220)
upper_white_filter2 = (255, 100, 255)

lower_white_filter3 = (0, 0, 240)
upper_white_filter3 = (255, 100, 255)


def convertToHSV(image, lower_white, upper_white):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_image, lower_white, upper_white)
    white_parts = cv.bitwise_and(image, image, mask=mask)

    return white_parts


def findBoardCorners(image):
    #  am aplicat cateva filtre pentru detectia colturilor tablei de joc
    gray_image = cv.cvtColor(convertToHSV(image, lower_white_board, upper_white_board), cv.COLOR_BGR2GRAY)
    # show_image('', gray_image)
    image_m_blur = cv.medianBlur(gray_image, 11)
    # show_image('medianBlur', image_m_blur)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 11)
    # show_image('GaussianBlur', image_g_blur)
    image_sharpened = cv.addWeighted(image_m_blur, 0.9, image_g_blur, -0.8, 0)
    # show_image('image_sharpened', image_sharpened)
    _, thresh = cv.threshold(image_sharpened, 1, 255, cv.THRESH_BINARY)
    # show_image('thresh', thresh)
    kernel = np.ones((19, 19), np.uint8)
    thresh = cv.dilate(thresh, kernel)

    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    #  cod preluat de la laborator
    for i in range(len(contours)):
        if (len(contours[i]) > 3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis=1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]

            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    image_copy = cv.cvtColor(gray_image.copy(), cv.COLOR_GRAY2BGR)
    cv.circle(image_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(image_copy, tuple(bottom_right), 20, (0, 0, 255), -1)

    # show_image("detected corners", image_copy)

    return top_left, top_right, bottom_left, bottom_right


def extract_board(image, top_left, top_right, bottom_left, bottom_right):

    # tabla 15x15 -> 2250 - multiplu de 15
    width_board = 2250
    height_board = 2250

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    destination_of_puzzle = np.array([[0, 0], [width_board, 0], [width_board, height_board], [0, height_board]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)
    result = cv.warpPerspective(image, M, (width_board, height_board))

    return result


dist_linii_caroiaj = 147

lines_horizontal = []
for i in range(20, 2231, dist_linii_caroiaj):
    l = []
    l.append((20, i))
    l.append((2231, i))
    lines_horizontal.append(l)

lines_vertical = []
for i in range(20, 2231, dist_linii_caroiaj):
    l = []
    l.append((i, 20))
    l.append((i, 2231))
    lines_vertical.append(l)


#  se presupune ca imaginea este in format BGR (color)
def vizualizare_caroiaj(image):
    for line in lines_vertical:
        cv.line(image, line[0], line[1], (0, 255, 0), 5)
    for line in lines_horizontal:
        cv.line(image, line[0], line[1], (0, 0, 255), 5)
    show_image('caroiaj', image)


litere = [chr(i) for i in range(ord('A'), ord('O') + 1)]

def litera_pentru_index(index):
    if 0 <= index <= len(litere) - 1:
        return litere[index]
    else:
        return 'index out of bound'


#  cele 2 imagini trebuie sa fie in format grayscale
def comparePatches(image1, image2):
    histSize = [2]
    channels = [0]
    intensity_ranges = [0, 256]

    hist_img1 = cv.calcHist([image1], channels, None, histSize, intensity_ranges)
    cv.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    hist_img2 = cv.calcHist([image2], channels, None, histSize, intensity_ranges)
    cv.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    return cv.compareHist(hist_img1, hist_img2, 0)


def distanta_manhattan(x1, y1, x2, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)


#  functia are ca scop eliminarea valorilor fals pozitive grosolane (dist. manhattan != 1)
def proceseazaPredictii(multime_puncte):
    list = []
    if len(multime_puncte) > 2:
        puncte = []
        for i in range(len(multime_puncte)):
            for j in range(i + 1, len(multime_puncte)):
                distanta = distanta_manhattan(multime_puncte[i][0], multime_puncte[i][1], multime_puncte[j][0], multime_puncte[j][1])
                if distanta == 1:
                    puncte.append((multime_puncte[i], multime_puncte[j]))

        list.append(puncte[0][0])
        list.append(puncte[0][1])
        return list  # iau la intamplare, nu tratez cazuri particulare (e nevoie de valoarea capetelor, pozitii etc)

    return multime_puncte


def gasestePiesa(image1, image2, matrix):

    image1_01 = cv.cvtColor(convertToHSV(image1, lower_white_filter1, upper_white_filter1), cv.COLOR_BGR2GRAY)
    image1_02 = cv.cvtColor(convertToHSV(image1, lower_white_filter2, upper_white_filter2), cv.COLOR_BGR2GRAY)
    image1_03 = cv.cvtColor(convertToHSV(image1, lower_white_filter3, upper_white_filter3), cv.COLOR_BGR2GRAY)

    image2_01 = cv.cvtColor(convertToHSV(image2, lower_white_filter1, upper_white_filter1), cv.COLOR_BGR2GRAY)
    image2_02 = cv.cvtColor(convertToHSV(image2, lower_white_filter2, upper_white_filter2), cv.COLOR_BGR2GRAY)
    image2_03 = cv.cvtColor(convertToHSV(image2, lower_white_filter3, upper_white_filter3), cv.COLOR_BGR2GRAY)

    prag = 0.55
    list = []
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):

            if matrix[i, j] != 0:
                continue

            y_min = lines_vertical[j][0][0] + int(dist_linii_caroiaj/10)
            y_max = lines_vertical[j + 1][1][0] - int(dist_linii_caroiaj/10)
            x_min = lines_horizontal[i][0][1] + int(dist_linii_caroiaj/10)
            x_max = lines_horizontal[i + 1][1][1] - int(dist_linii_caroiaj/10)

            patch1_01 = image1_01[x_min:x_max, y_min:y_max].copy()
            patch1_02 = image1_02[x_min:x_max, y_min:y_max].copy()
            patch1_03 = image1_03[x_min:x_max, y_min:y_max].copy()

            patch2_01 = image2_01[x_min:x_max, y_min:y_max].copy()
            patch2_02 = image2_02[x_min:x_max, y_min:y_max].copy()
            patch2_03 = image2_03[x_min:x_max, y_min:y_max].copy()

            coef = []
            coef.append(comparePatches(patch1_01, patch2_01))
            coef.append(comparePatches(patch1_02, patch2_02))
            coef.append(comparePatches(patch1_03, patch2_03))

            nr_voturi = 0
            for c in coef:
                if c < prag:
                    nr_voturi += 1

            if nr_voturi >= 2:
                list.append((i+1, j+1))

    return list


def findCircles(image):
    gray = image.copy()
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    gray_blurred = cv.GaussianBlur(gray, (0, 0), 2)

    circles = cv.HoughCircles(
        gray_blurred,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=10,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )

    nr_points1 = 0
    nr_points2 = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(image, center, 1, (255, 0, 0), 3)
            cv.circle(image, center, i[2], (255, 0, 255), 3)

        # calculeaza aprox pozitia linie din mijloc
        h, w, _ = np.array(image).shape
        x_middle = []
        y_middle = []
        if h > w:
            x_middle.append(0)
            x_middle.append(w)
            y_middle.append(h // 2)
            y_middle.append(h // 2)
            cv.line(image, (0, h // 2), (w, h // 2), (0, 255, 0), 2)
        else:
            x_middle.append(w // 2)
            x_middle.append(w // 2)
            y_middle.append(0)
            y_middle.append(h)
            cv.line(image, (w // 2, 0), (w // 2, h), (0, 255, 0), 2)

        middle = tuple(zip(x_middle, y_middle))

        for circle in circles[0]:
            if h > w:
                if circle[1] < middle[0][1]:
                    nr_points1 += 1
                else:
                    nr_points2 += 1
            else:
                if circle[0] < middle[0][0]:
                    nr_points1 += 1
                else:
                    nr_points2 += 1

    # show_image('circles', image)

    return nr_points1, nr_points2


# un domino poate atinge maxim o stea la o mutare
def returneazaPunctaj(tabla, puncte):
    for punct in puncte:
        if tabla[punct[0] - 1][punct[1] - 1] != 0:
            return tabla[punct[0] - 1][punct[1] - 1]

    return 0


if __name__ == '__main__':

# -------------------------------------------  Configurare joc ---------------------------------------------------------
    #  setare path-uri
    path_tabla_de_joc = "C:/Users/Antonio/Desktop/TEMA1_CAVA/imagini_auxiliare/01.jpg"
    path_seturi_date = "C:/Users/Antonio/Desktop/TEMA1_CAVA/antrenare/"
    # path_seturi_date = "C:/Users/Antonio/Desktop/TEMA1_CAVA/evaluare/fake_test/"
    path_predictii = "C:/Users/Antonio/Desktop/TEMA1_CAVA/predictii/"

    nr_jocuri = 5  #  nr jocuri
    nr_runde_joc = 20  #  nr runde per joc
    bonus = 3

    #  tabla de joc hardcodata
    tabla = [
        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5],
        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 4, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, 4, 0],
        [0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 0],
        [4, 0, 0, 3, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 4],
        [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0],
        [0, 0, 3, 0, 0, 4, 0, 0, 0, 4, 0, 0, 3, 0, 0],
        [5, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 5]
    ]

    #  drumul jucatorilor hardcodat
    drum = [1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1,
            5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2,
            1, 2, 5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6,
            4, 4, 1, 6, 6, 3, 0]

# -------------------------------------------  Algoritm ----------------------------------------------------------------

    #  citire imagine tabla de joc
    full_board = cv.imread(path_tabla_de_joc)
    top_left, top_right, bottom_left, bottom_right = findBoardCorners(full_board)
    board = extract_board(full_board, top_left, top_right, bottom_left, bottom_right)


    for joc in range(1, nr_jocuri + 1):  # pentru o runda schimba in range(1,2), altfel range(1,6)

        matrix = np.zeros((15, 15), dtype='uint8')  # pozitiile pieselor detectate
        cnt = 0  # nr piesei detectate
        prev_image = board  # imaginea precendenta cu care compar

        poz_jucatori = [-1, -1]  # pozitia jucatorilor pentru fiecare joc
        mutari = []  # mutarile aferente jocului

        fisierMutari = path_seturi_date + str(joc) + '_mutari.txt'

        with open(fisierMutari, 'r') as fisier:
            linie = fisier.readline().strip()
            while linie:
                mutari.append(int(linie[-1:]))
                linie = fisier.readline().strip()


        #  citire date
        files = os.listdir(path_seturi_date)
        for file in files:
            if file.startswith(str(joc)) and file[-3:] == 'jpg':
                image = cv.imread(path_seturi_date + file)
                scor_acumulat_per_runda = [0, 0]
                nr_points1_hough = 0; nr_points2_hough = 0; turn = 1
                if cnt == 0:
                    top_left, top_right, bottom_left, bottom_right = findBoardCorners(image)
                current_image = extract_board(image, top_left, top_right, bottom_left, bottom_right)
                # show_image('current_image', current_image)

#----------------------------------------------  task-ul 1 -------------------------------------------------------------

                pozitii_detectate = gasestePiesa(prev_image, current_image, matrix)
                pozitii_postprocesate = proceseazaPredictii(pozitii_detectate)
                cnt += 1
                for poz in pozitii_postprocesate:
                    matrix[poz[0] - 1, poz[1] - 1] = cnt
                prev_image = current_image
                print('Pozitii detectate: ', pozitii_postprocesate)

# ----------------------------------------------  task-ul 2 ------------------------------------------------------------

                if len(pozitii_postprocesate) == 2:

                    # scadem cate un 1 pt ca lucram cu matricea aici, nu afisam
                    i1 = pozitii_postprocesate[0][0] - 1
                    j1 = pozitii_postprocesate[0][1] - 1

                    # setez dimensiunea patch-ului in functie de pozitia la care a fost gasit
                    if j1 > 0:
                        y1_min = lines_vertical[j1][0][0] - 10
                    else:
                        y1_min = lines_vertical[j1][0][0] - 18

                    if j1 != 14:
                        y1_max = lines_vertical[j1 + 1][1][0] + 10
                    else:
                        y1_max = lines_vertical[j1 + 1][1][0] + 18

                    if i1 > 0:
                        x1_min = lines_horizontal[i1][0][1] - 10
                    else:
                        x1_min = lines_horizontal[i1][0][1] - 18

                    if i1 != 14:
                        x1_max = lines_horizontal[i1 + 1][1][1] + 10
                    else:
                        x1_max = lines_horizontal[i1 + 1][1][1] + 18


                    # scadem cate un 1 pt ca lucram cu matricea aici, nu afisam
                    i2 = pozitii_postprocesate[1][0] - 1
                    j2 = pozitii_postprocesate[1][1] - 1


                    # setez dimensiunea patch-ului in functie de pozitia la care a fost gasit
                    if j2 > 0:
                        y2_min = lines_vertical[j2][0][0] - 10
                    else:
                        y2_min = lines_vertical[j2][0][0] - 18

                    if j2 != 14:
                        y2_max = lines_vertical[j2 + 1][1][0] + 10
                    else:
                        y2_max = lines_vertical[j2 + 1][1][0] + 18

                    if i2 > 0:
                        x2_min = lines_horizontal[i2][0][1] - 10
                    else:
                        x2_min = lines_horizontal[i2][0][1] - 18

                    if i2 != 14:
                        x2_max = lines_horizontal[i2 + 1][1][1] + 10
                    else:
                        x2_max = lines_horizontal[i2 + 1][1][1] + 18


                    hsv_patch = convertToHSV(current_image[x1_min:x2_max, y1_min:y2_max], lower_white_filter2, upper_white_filter2)
                    patch_for_Hough = cv.cvtColor(cv.medianBlur(hsv_patch, 3), cv.COLOR_BGR2GRAY)
                    # show_image('patch', hsv_patch)

                    nr_points1_hough, nr_points2_hough = findCircles(patch_for_Hough)
                    print('Hough: ', nr_points1_hough, nr_points2_hough)

# ----------------------------------------------  task-ul 3 ------------------------------------------------------------

                    turn = mutari[cnt-1]  # al cui e randul sa mute

                    scor_nimerit = returneazaPunctaj(tabla, pozitii_postprocesate)

                    #  cat a avansat din drum fiecare jucator
                    if poz_jucatori[0] == -1:
                        piesa_pe_care_sta_juc1 = -1
                    else:
                        piesa_pe_care_sta_juc1 = drum[poz_jucatori[0]]

                    if poz_jucatori[1] == -1:
                        piesa_pe_care_sta_juc2 = -1
                    else:
                        piesa_pe_care_sta_juc2 = drum[poz_jucatori[1]]


                    #  piesa dubla de domino => scor dublu
                    if nr_points1_hough == nr_points2_hough:
                        scor_acumulat_per_runda[turn - 1] += scor_nimerit * 2
                    else:
                        scor_acumulat_per_runda[turn - 1] += scor_nimerit


                    #  modul de acordare al bonusului
                    if piesa_pe_care_sta_juc1 == nr_points1_hough or piesa_pe_care_sta_juc1 == nr_points2_hough:
                        scor_acumulat_per_runda[0] += bonus
                    if piesa_pe_care_sta_juc2 == nr_points1_hough or piesa_pe_care_sta_juc2 == nr_points2_hough:
                        scor_acumulat_per_runda[1] += bonus

                    poz_jucatori[0] += scor_acumulat_per_runda[0]
                    poz_jucatori[1] += scor_acumulat_per_runda[1]

                    print('Scor: ', scor_acumulat_per_runda[turn - 1])


# -------------------------------------  formatare fisiere output-------------------------------------------------------

                # formatarea fisierelor de output
                if 1 <= cnt <= 9:
                    numar_fisier = '0' + str(cnt)
                else:
                    numar_fisier = str(cnt)
                nume_fisier = str(joc) + "_" + numar_fisier + ".txt"
                with open(path_predictii + nume_fisier, 'w') as fisier:
                    fisier.write(str(pozitii_postprocesate[0][0]) + litera_pentru_index(pozitii_postprocesate[0][1] - 1) + ' ' + str(nr_points1_hough) + '\n')
                    fisier.write(str(pozitii_postprocesate[1][0]) + litera_pentru_index(pozitii_postprocesate[1][1] - 1) + ' ' + str(nr_points2_hough) + '\n')
                    fisier.write(str(scor_acumulat_per_runda[turn - 1]))
