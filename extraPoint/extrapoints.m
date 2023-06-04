clear;
clc;

a1=imread('book_size.jpg');
a2=imread('tom_computer.png'); % double plays a big role
a=get(0);
figure('position',a.MonitorPositions);
subplot(1,2,1);imshow(a1);
subplot(1,2,2);imshow(a2);

num_points = 4;
[x,y] = ginput(num_points);
close all;
% reshape(x(1:8),1,8),ones(1,8);
point_ref = [reshape(x(1:num_points),1,num_points);reshape(y(1:num_points),1,num_points);ones(1,num_points)];
[h, w,~] = size(a1);
point1=[1,1,1];
point2=[w,1,1];
point3=[w,h,1];
point4=[1,h,1];
point_src=[point1;point2;point3;point4]';

hh = getHomographyMatrix(point_ref,point_src,num_points);

[height_wrap, width_wrap,~] = size(a1);
[height_unwrap, width_unwrap,~] = size(a2);

[newH, newW, newX, newY, xB, yB] = getNewSize(hh, height_wrap, width_wrap, height_unwrap, width_unwrap);

[X,Y] = meshgrid(1:width_wrap,1:height_wrap);
[XX,YY] = meshgrid(newX:newX+newW-1, newY:newY+newH-1);
AA = ones(3,newH*newW);
AA(1,:) = reshape(XX,1,newH*newW);
AA(2,:) = reshape(YY,1,newH*newW);

AA = hh*AA;
XX = reshape(AA(1,:)./AA(3,:), newH, newW);
YY = reshape(AA(2,:)./AA(3,:), newH, newW);

% INTERPOLATION, WARP IMAGE A INTO NEW IMAGE
newImage(:,:,1) = interp2(X, Y, double(a1(:,:,1)), XX, YY);
newImage(:,:,2) = interp2(X, Y, double(a1(:,:,2)), XX, YY);
newImage(:,:,3) = interp2(X, Y, double(a1(:,:,3)), XX, YY);

% BLEND IMAGE BY CROSS DISSOLVE
[newImage] = blend(newImage, a2, xB, yB);
figure('position',a.MonitorPositions);
subplot(1,3,1);imshow(a1);
subplot(1,3,2);imshow(a2);
subplot(1,3,3);imshow(uint8(newImage));
%%
function [hh] = getHomographyMatrix(point_ref, point_src, npoints)
% Use corresponding points in both images to recover the parameters of the transformation 
% Input:
% x_ref, x_src --- x coordinates of point correspondences
% y_ref, y_src --- y coordinates of point correspondences
% Output:
% h --- matrix of transformation

% NUMBER OF POINT CORRESPONDENCES
x_ref = point_ref(1,:)';
y_ref = point_ref(2,:)';
x_src = point_src(1,:)';
y_src = point_src(2,:)';

% COEFFICIENTS ON THE RIGHT SIDE OF LINEAR EQUATIONS
A = zeros(npoints*2,8);
A(1:2:end,1:3) = [x_ref, y_ref, ones(npoints,1)];
A(2:2:end,4:6) = [x_ref, y_ref, ones(npoints,1)];
A(1:2:end,7:8) = [-x_ref.*x_src, -y_ref.*x_src];
A(2:2:end,7:8) = [-x_ref.*y_src, -y_ref.*y_src];

% COEFFICIENT ON THE LEFT SIDE OF LINEAR EQUATIONS
B = [x_src, y_src];
B = reshape(B',npoints*2,1);

% SOLVE LINEAR EQUATIONS
h = A\B;

hh = [h(1),h(2),h(3);h(4),h(5),h(6);h(7),h(8),1];
end

%%
function [newH, newW, x1, y1, x2, y2] = getNewSize(transform, h2, w2, h1, w1)
% Calculate the size of new mosaic
% Input:
% transform - homography matrix
% h1 - height of the unwarped image
% w1 - width of the unwarped image
% h2 - height of the warped image
% w2 - height of the warped image
% Output:
% newH - height of the new image
% newW - width of the new image
% x1 - x coordate of lefttop corner of new image
% y1 - y coordate of lefttop corner of new image
% x2 - x coordate of lefttop corner of unwarped image
% y2 - y coordate of lefttop corner of unwarped image

% CREATE MESH-GRID FOR THE WARPED IMAGE
[X,Y] = meshgrid(1:w2,1:h2);
AA = ones(3,h2*w2);
AA(1,:) = reshape(X,1,h2*w2);
AA(2,:) = reshape(Y,1,h2*w2);

% DETERMINE THE FOUR CORNER OF NEW IMAGE
newAA = transform\AA;
new_left = fix(min([1,min(newAA(1,:)./newAA(3,:))]));
new_right = fix(max([w1,max(newAA(1,:)./newAA(3,:))]));
new_top = fix(min([1,min(newAA(2,:)./newAA(3,:))]));
new_bottom = fix(max([h1,max(newAA(2,:)./newAA(3,:))]));

newH = new_bottom - new_top + 1;
newW = new_right - new_left + 1;
x1 = new_left;
y1 = new_top;
x2 = 2 - new_left;
y2 = 2 - new_top;
end


%%
function [newImage] = blend(warped_image, unwarped_image, x, y)
% Blend two image by using cross dissolve
% Input:
% warped_image - original image
% unwarped_image - the other image
% x - x coordinate of the lefttop corner of unwarped image
% y - y coordinate of the lefttop corner of unwarped image
% Output:
% newImage

% MAKE MASKS FOR BOTH IMAGES 
warped_image(isnan(warped_image))=0;
maskA = (warped_image(:,:,1)>0 |warped_image(:,:,2)>0 | warped_image(:,:,3)>0);
newImage = zeros(size(warped_image));
newImage(y:y+size(unwarped_image,1)-1, x: x+size(unwarped_image,2)-1,:) = unwarped_image;

% overlap part paint black
newImage(:,:,1) = uint8(newImage(:,:,1)- 1000*warped_image(:,:,1));
newImage(:,:,2) = uint8(newImage(:,:,2)- 1000*warped_image(:,:,2));
newImage(:,:,3) = uint8(newImage(:,:,3)- 1000*warped_image(:,:,3));

newImage(:,:,1) = warped_image(:,:,1) + newImage(:,:,1);
newImage(:,:,2) = warped_image(:,:,2) + newImage(:,:,2);
newImage(:,:,3) = warped_image(:,:,3) + newImage(:,:,3);

end