import numpy as np
np.set_printoptions(threshold=np.inf)
import os
from scipy import signal
import cv2
import glob
import random
import matplotlib.pyplot as plt

######### obtain the image
#filenames = [img for img in glob.glob("human/*.jpg")]
#filenames.sort()
# for img in filenames:
#     n= cv2.imread(img)
# # obtain the gray scale
#     gray = cv2.cvtColor(n,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('vid',gray)
#     cv2.waitKey(0)
#


###### created car template
# img = cv2.imread('car/frame0020.jpg')
# # cv2.imshow('img',img)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #cv2.imshow('img',gray)
# cropped = gray[100:275, 125:335]
# #cv2.imshow('img',cropped)
# #cv2.imwrite('templates/car_template.jpg', cropped)
# cv2.waitKey(0)

### points = [[x1,y1],[x2,y2]]  diagonally corner points of the template
###### created human template
# img = cv2.imread('human/0148.jpg')
# # cv2.imshow('img',img)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #cv2.imshow('img',gray)
# cropped = gray[290:360, 250:288]
# #cv2.imshow('img',cropped)
# cv2.imwrite('templates/human_template.jpg', cropped)
# cv2.waitKey(0)

###### created vase template
# img = cv2.imread('vase/0019.jpg')
# #cv2.imshow('img',img)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #cv2.imshow('img',gray)
# cropped = gray[72:148, 123:172]
# #cv2.imshow('img',cropped)
# cv2.imwrite('templates/vase_template.jpg', cropped)
# cv2.waitKey(0)


# obtain the warped image and warped parameters

# img = cv2.imread('car/frame0020.jpg')
# cv2.imshow('img',img)
##### Image gradients

#grad_x, grad_y = image_gradients(img)
# cv2.imshow('x',grad_x)
# cv2.imshow('y',grad_y)
# cv2.waitKey(0)


#### Lucas kanade algorithm to find optical flow vector values u and v


# print(gray2.shape)
# print(gray1.shape)


# def optical_flow(img1, img2, win_size, thresh = 1e-2):
#
#     kernel_x = np.array([[-1., 1.], [-1., 1.]])
#     kernel_y = np.array([[-1., -1.], [1., 1.]])
#     kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
#
#     # print(kernel_x)
#     # print(kernel_y)
#     # print(kernel_t)
#
#     win = int(win_size/2)
#
#     img1 = img1/255.  # normalizing
#     img2 = img2/255.  # normalizing
#
#
#     fx = cv2.filter2D(src=img1,ddepth = -1,kernel=kernel_x)
#     fy = cv2.filter2D(src=img2,ddepth = -1,kernel=kernel_y)
#     ft = cv2.filter2D(src=img2,ddepth = -1,kernel=kernel_t) + cv2.filter2D(src=img1,ddepth = -1,kernel=-kernel_t)
#
#     # cv2.imshow('fx',fx)
#     # cv2.imshow('fy',fy)
#     # cv2.imshow('ft',ft)
#
#     u = np.zeros(img1.shape)
#     v = np.zeros(img2.shape)
#
#     # print(img1.shape)
#     # print('\n',img1.shape)
#
#     # print('u=',u)
#     # print('v=',v)
#
#     # cv2.waitKey(0)
#
#     for i in range(win,img1.shape[0]-win):
#         for j in range(win,img1.shape[1]-win):
#             I_x = fx[i-win:i+win+1, j-win:j+win+1].flatten()
#             I_y = fy[i-win:i+win+1, j-win:j+win+1].flatten()
#             I_t = ft[i-win:i+win+1, j-win:j+win+1].flatten()
#             # b = np.reshape(I_t, (I_t.shape[0],1))
#             # A = np.vstack((I_x, I_y)).T
#
#             # if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= thresh:
#             #      nu = np.matmul(np.linalg.pinv(A), b)
#             #
#             #      nu =
#             #
#             #      u[i,j]=nu[0]
#             #      v[i,j]=nu[1]
#     # A = np.vstack((I_x,I_y)).T
#     # B = np.reshape(I_t,(I_t.shape[0],1))
#     #
#     # print('A=',A,'\n')
#
#
#             # y = np.linalg.inv(np.matmul(A.T,A))
#
#
#             # print(abs(np.linalg.eigvals(np.matmul(A.T,A))))
#
#             # print(np.linalg.eigvals(y))
#             # print(np.linalg.eigvals(np.matmul(A.T,A)))
#
#
#             # z = np.matmul(A.T,B)
#             #
#             # # x = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,B))
#             #
#             # x = np.matmul(y,z)
#             #
#             # # print(x)
#             #
#             # u[i,j] = x[0]
#             # v[i,j] = x[1]
#
#
#
#             # print(x)
#
#             # print(u)
#             # print(v)
#
#             #
#             # if np.min(abs(np.linalg.eigvals(np.matmul(A.T,A))))>= thresh:
#             #     #print('hello')
#             #     nu = np.matmul(np.linalg.pinv(A),B)
#             #
#             #     print(nu.shape)
#             #     # u[i,j] = nu[0]
#             #     # v[i,j] = nu[1]
#
#     A = np.vstack((I_x,I_y)).T
#     B = np.reshape(I_t,(I_t.shape[0],1))
#
#     # print(A)
#
#     # print(B)
#
#     x = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,B))
#
#     print(x)
#
#     u = x[0]
#     v = x[1]
#
#     # print(I_x.shape)
#     # print(I_y.shape)
#     # print(I_t.shape)
#
#     # A x = B
#     #
#     # A = [ix iy]
#     #
#     # b = [it]
#     #
#     # x = [u;v]
#
#     # print('A=',A.shape)
#     # print('B=',B.shape)
#     #
#     # print('done')
#
#     return u,v


# c = template_create(gray1,points)

# cv2.imshow('temp',c)
# cv2.waitKey(0)



# vel_u , vel_v = optical_flow(gray1,gray2,21)
#
# plt.quiver(vel_u, vel_v,scale=100)#,label='eigen vector1')
#
# plt.show()
# cv2.imshow('u',vel_u)
# cv2.waitKey(0)
#
# print(vel_u.shape)
# print('v=',vel_v.shape)



# img1 = cv2.imread('car/frame0020.jpg')  ### image at time step 1
# img2 = cv2.imread('car/frame0021.jpg')  ### image at time step 2
#
# gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#
# # points = [[125,100],[335,275]]
# points = [[120,90],[340,280]]

#print(points[1][0])

def warp_optical_flow(img1, img2):#, win_size, thresh = 1e-2):

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25

    # print(kernel_x)
    # print(kernel_y)
    # print(kernel_t)

    # win = int(win_size/2)

    # img1 = img1/255.  # normalizing
    # img2 = img2/255.  # normalizing


    fx = cv2.filter2D(src=img1,ddepth = -1,kernel=kernel_x)
    fy = cv2.filter2D(src=img1,ddepth = -1,kernel=kernel_y)
    ft = cv2.filter2D(src=img2,ddepth = -1,kernel=kernel_t) + cv2.filter2D(src=img1,ddepth = -1,kernel=-kernel_t)

    # fx = signal.convolve2d(img1, kernel_x, boundary='symm', mode='same')
    # fy = signal.convolve2d(img1, kernel_y, boundary='symm', mode='same')
    # ft = signal.convolve2d(img2, kernel_t, boundary='symm', mode='same') + signal.convolve2d(img1, -kernel_t, boundary='symm', mode='same')


    features = cv2.goodFeaturesToTrack(img1,1000,0.01,10)

    feature = np.int0(features)

    # cv2.imshow('fx',fx)
    # cv2.imshow('fy',fy)
    # cv2.imshow('ft',ft)

    u = np.nan*np.ones(img1.shape)
    v = np.nan*np.ones(img1.shape)


    for l in feature:
        j,i = l.ravel()
        # calculating the derivatives for the neighbouring pixels
        # since we are using  a 3*3 window, we have 9 elements for each derivative.

        IX = ([fx[i-1,j-1],fx[i,j-1],fx[i-1,j-1],fx[i-1,j],fx[i,j],fx[i+1,j],fx[i-1,j+1],fx[i,j+1],fx[i+1,j-1]]) #The x-component of the gradient vector
        IY = ([fy[i-1,j-1],fy[i,j-1],fy[i-1,j-1],fy[i-1,j],fy[i,j],fy[i+1,j],fy[i-1,j+1],fy[i,j+1],fy[i+1,j-1]]) #The Y-component of the gradient vector
        IT = ([ft[i-1,j-1],ft[i,j-1],ft[i-1,j-1],ft[i-1,j],ft[i,j],ft[i+1,j],ft[i-1,j+1],ft[i,j+1],ft[i+1,j-1]]) #The XY-component of the gradient vector

        # Using the minimum least squares solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK)) # transpose of A
        LK = np.array(np.matrix.transpose(LK))

        A1 = np.dot(LK_T,LK) #Psedudo Inverse
        A2 = np.linalg.pinv(A1)
        A3 = np.dot(A2,LK_T)

        (u[i,j],v[i,j]) = np.dot(A3,IT) # we have the vectors with minimized square error

    # print(img1.shape)
    # print('\n',img1.shape)

    # print('u=',u)
    # print('v=',v)

    # cv2.waitKey(0)
    #
    # for i in range(win,img1.shape[0]-win):
    #     for j in range(win,img1.shape[1]-win):
    #         I_x = fx[i-win:i+win+1, j-win:j+win+1].flatten()
    #         I_y = fy[i-win:i+win+1, j-win:j+win+1].flatten()
    #         I_t = ft[i-win:i+win+1, j-win:j+win+1].flatten()
    #         b = np.reshape(I_t, (I_t.shape[0],1))
    #         A = np.vstack((I_x, I_y)).T
    #
    #         if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= thresh:
    #              nu = np.matmul(np.linalg.pinv(A), b)
    #              u[i,j]=nu[0]
    #              v[i,j]=nu[1]
    # # A = np.vstack((I_x,I_y)).T
    # # B = np.reshape(I_t,(I_t.shape[0],1))
    # #
    # # print('A=',A,'\n')
    # #
    # #
    # #         y = np.linalg.inv(np.matmul(A.T,A))
    # #
    # #
    # #         print(abs(np.linalg.eigvals(np.matmul(A.T,A))))
    # #
    # #         print(np.linalg.eigvals(y))
    # #         print(np.linalg.eigvals(np.matmul(A.T,A)))
    # #
    # #
    # #         z = np.matmul(A.T,B)
    # #
    # #         # x = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,B))
    # #
    # #         x = np.matmul(y,z)
    # #
    # #         # print(x)
    # #
    # #         u[i,j] = x[0]
    # #         v[i,j] = x[1]
    # #
    # #
    # #
    # #         print(x)
    # #
    # #         print(u)
    # #         print(v)
    # #
    # #
    # #         if np.min(abs(np.linalg.eigvals(np.matmul(A.T,A))))>= thresh:
    # #             #print('hello')
    # #             nu = np.matmul(np.linalg.pinv(A),B)
    # #
    # #             print(nu.shape)
    # #             # u[i,j] = nu[0]
    # #             # v[i,j] = nu[1]
    # #
    # # A = np.vstack((I_x,I_y)).T
    # # B = np.reshape(I_t,(I_t.shape[0],1))
    # #
    # # # print(A)
    # #
    # # # print(B)
    # #
    # # x = np.matmul(np.linalg.inv(np.matmul(A.T,A)),np.matmul(A.T,B))
    # #
    # # print(x)
    # #
    # # u = x[0]
    # # v = x[1]
    # #
    # # # print(I_x.shape)
    # # # print(I_y.shape)
    # # # print(I_t.shape)
    # #
    # # # A x = B
    # # #
    # # # A = [ix iy]
    # # #
    # # # b = [it]
    # # #
    # # # x = [u;v]
    # #
    # # # print('A=',A.shape)
    # # # print('B=',B.shape)
    # # #
    # # # print('done')

    return u,v

def template_create(gray_img,points):
    ### crop = img[y1:y2, x1:x2]
    cropped = gray_img[points[0][1]:points[1][1], points[0][0]:points[1][0]]
    return cropped

def warp_paramters(P,points):

    # print(P)
    p1 = P[0]
    p2 = P[1]
    p3 = P[2]
    p4 = P[3]
    p5 = P[4]
    p6 = P[5]

    x_start = points[0][0]
    x_end = points[1][0]
    y_start = points[0][1]
    y_end = points[1][1]

    w = np.reshape(np.array([[1+p1,p3,p5],[p2,1+p4,p6]]),(2,3))
    # print(w)

    warp_mat = []
    warp_full = []

    for i in range(x_start,x_end):
        # print(i)
        for j in range(y_start,y_end):
            # print(j)
            # print('\n')
            mul = np.matmul(w,np.array([[i],[j],[1]]))
            # Warp_mat.append()
            warp_mat.append(tuple((mul[0][0],mul[1][0])))

    return warp_mat

def warped_image(warp_mat,frame,points):


    # print('frame in warped image funciton',frame.shape)

    x_start = points[0][0]
    x_end = points[1][0]
    y_start = points[0][1]
    y_end = points[1][1]

    # print(x_end-x_start)

    I = np.zeros([points[1][1]-points[0][1],points[1][0]-points[0][0]])
    # print(I.shape)
    t = 0


    for i in range(0,x_end-x_start):

        for j in range(0,y_end-y_start):
            # print(int(warp_mat[t][0]),int(warp_mat[t][1]))
            # I[j,i] = frame[int(warp_mat[t][0]),int(warp_mat[t][1])]
            # print(warp_mat[t][0],warp_mat[t][1])
            I[j,i] = frame[int(warp_mat[t][0]),int(warp_mat[t][1])]
            t+=1
            # print(t)

    # print(I.shape)

    # cv2.imshow('img',I)
    # cv2.waitKey(0)

    return I

# remove the template from the warped image to obtain the error
def error(template,I):

    # x_start = points[0][0]
    # x_end = points[1][0]
    # y_start = points[0][1]
    # y_end = points[1][1]
    #
    # error = np.abs(frame[y_start:y_end,x_start:x_end]-I)

    error = np.abs(template-I)

    # cv2.imshow('inmg',error)
    # cv2.waitKey(0)

    return error

def image_gradients(img):
    # obtain the gradient X
    #CV_64F  CV_8U
    gradient_sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)
    # abs_sobelx_64f = np.absolute(gradient_sobelx)
    # sobelx_8u = np.uint8(abs_sobelx_64f)
    # cv2.imshow('sob',sobelx_8u )

    # obtain the gradient Y
    gradient_sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)
    # abs_sobely_64f = np.absolute(gradient_sobely)
    # sobely_8u = np.uint8(abs_sobely_64f)
    # cv2.imshow('sob y',sobely_8u )
    # cv2.waitKey(0)
    # return sobelx_8u,sobely_8u

    return gradient_sobelx,gradient_sobely

# obtian the Jacobian matrix
def jacobian(x,y):
    # dw_dp = np.array(np.mat('x 0 y 0 1 0;0 x 0 y 0 1'))
    dw_dp = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
    # print(dw_dp.shape)
    return dw_dp

### obtaining the gradient croppes images
# grad_x_cropped = template_create(grad_x,points)
# grad_y_cropped = template_create(grad_y,points)
# cv2.imshow('grad',grad_y_cropped)
# cv2.waitKey(0)

def iterative(gray1,points,T,P):

    # P = np.array([[0.00078066],[0.00145158],[0.00640321],[0.00894728],[0.10726949],[0.35408914]])

    iter = 0

    norm = 0

    # while(iter<10):
    while(norm < 0.010):


        # print('iter',iter)
        warp_params =  warp_paramters(P,points)

        warped_im = warped_image(warp_params,gray1,points)

        #cv2.imshow('waaaaaa',warped_im)

        # print(len(warp_params))

        grad_x , grad_y = image_gradients(gray1)

        # cv2.imshow('x',grad_x)
        # print(grad_x.shape)
        # print(grad_y.shape)

        ### obtaining the gradients of cropped images in x and y
        warp_grad_x  = warped_image(warp_params,grad_x,points)
        warp_grad_y  = warped_image(warp_params,grad_y,points)
        # warp_grad_x_y = warp_grad_x+warp_grad_y

        ## stacking the gradients
        stacked_warped_grad  = np.dstack((warp_grad_x,warp_grad_y))
        # print(stacked_warped_grad.shape)

        ## computing the Jacobian
        jac = jacobian(1,1)

        # using the Jacobian on warped gradients of X and warped Y and obtain the steepest gradient descent
        ### computing steepest descent
        steepest_descent = np.dot(stacked_warped_grad,jac)
        # print(steepest_descent.shape)

        steepest_descent_reshaped = np.reshape(steepest_descent,(steepest_descent.shape[0]*steepest_descent.shape[1],6))
        # print(steepest_descent_reshaped.shape)

        # applying the hessian to the steepest gradient descent
        ## computing the Hessian
        hessian = np.dot(steepest_descent_reshaped.T,steepest_descent_reshaped)
        # print(hessian)

        # inverse_hessian = np.linalg.inv(hessian)
        template = template_create(gray1,points)


        error_image = error(T,warped_im)

        # cv2.imshow('err',error_image)

        error_image_reshaped = np.reshape(error_image,(error_image.shape[0]*error_image.shape[1],1))
        # print(error_image_reshaped.shape)

        ### applying the inverse hessian to the "hessian of steepest gradient descent"
        delta_p =  np.matmul(np.linalg.pinv(hessian), np.matmul(steepest_descent_reshaped.T,error_image_reshaped))

        norm = np.linalg.norm(delta_p)
        print(np.linalg.norm(delta_p),iter)

        # print(delta_p)

        ### updating the P
        P = P + delta_p
        cv2.waitKey(5)
        iter+=1

    return P

img1 = cv2.imread('car/frame0020.jpg')  ### image at time step 1
# img2 = cv2.imread('car/frame0021.jpg')  ### image at time step 2

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

points = [[125,100],[335,275]]
# points = [[123,95],[338,280]]
P = np.array([[0],[0],[0],[0],[0],[0]])

template_image = template_create(gray1,points)
all_frames = []
######## obtain the image
filenames = [img for img in glob.glob("car/*.jpg")]
filenames.sort()
for img in filenames:
    n = cv2.imread(img)
    # obtain the gray scale
    gray1 = cv2.cvtColor(n,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('vid',gray1)
    # cv2.waitKey(0)

    P = iterative(gray1,points,template_image,P)
    # print(new_P)

    p1,p2,p3,p4,p5,p6 = P[0,0], P[1,0], P[2,0], P[3,0], P[4,0], P[5,0]

    W = np.matrix([[1+p1, p3, p5],[p2, 1+p4, p6]])
    # print(W)

    new_mat_1 = np.array([[points[0][0]],[points[0][1]],[1]])
    new_mat_2 = np.array([[points[1][0]],[points[1][1]],[1]])
    #
    wx1 = np.matmul(W,new_mat_1)
    wx2 = np.matmul(W,new_mat_2)
    # cv2.rectangle(gray1,points[0],points[1])
    cv2.rectangle(gray1,(wx1[0],wx1[1]),(wx2[0],wx2[1]),color=255)
    all_frames.append(gray1)
    cv2.imshow('gray1',gray1)
    # cv2.imshow('gray2',gray1)
    cv2.waitKey(5)


# source = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output_car.avi', source, 5.0, (640, 480))
# for iter1 in all_frames:
#     out.write(iter1)
#     cv2.waitKey(5)
# out.release()
