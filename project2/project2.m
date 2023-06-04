%% main----------------------------------------------------------------------------------------------------

% Matching of feature points, mainly using the detection of harris corner points, match one-way matching function
% is suitable for images with white edges, because there is no limited range when windowing and filtering, try to ensure that the corners are not on the edge
clc,clear all;
a1=imread('DSC_0281.JPG');
a2=imread('DSC_0282.JPG'); % double plays a big role
[ysize, xsize] = size(a1);

i_fig=1; % figure number

[result1,cnt1,r1,c1]=harris(a1);% Corner detection, get the original focus position map result
[result2,cnt2,r2,c2]=harris(a2);
figure(i_fig);subplot(1,2,1);imshow(a1);hold on;plot(c1,r1,'g.');
subplot(1,2,2);imshow(a2);hold on;plot(c2,r2,'g.'); title('Corner plots of Figures 1 and 2');
i_fig=i_fig+1;

[res2,ans2]=match(a1,cnt1,r1,c1,a2,cnt2,r2,c2);% Starting from result1 to search for possible reach in result2
[r22,c22]=find(res2==1);
[m22,n22]=size(r22);
cnt22=m22;

[res1,ans1]=match(a2,cnt22,r22,c22,a1,cnt1,r1,c1);% reverse search res2--result1
res1=and(res1,result1);   % Guaranteed reverse matching against impossible points
[r11,c11]=find(res1==1);
[m11,n11]=size(r11);
cnt11=m11;

respic = [a1,a2];
[rw,~] = size(ans1);

figure(i_fig);
imshow(respic);hold on;plot(ans1(:,2)+512,ans1(:,1),'rx');plot(ans1(:,4),ans1(:,3),'gx');
i_fig=i_fig+1;

for i=1:rw
%     if ans1(i, 5)>0.985  %Set thresholds for corresponding points to remove wrong pairs
        line([ans1(i,2)+512,ans1(i,4)],[ans1(i,1),ans1(i,3)])
%     end
end
title('The resulting matching corners');

ans1 = remove_same_pairs(ans1);
ans1 = remove_false(ans1, xsize*ysize);
[rw,D] = size(ans1);

figure(i_fig);
imshow(respic);hold on;plot(ans1(:,2)+512,ans1(:,1),'rx');plot(ans1(:,4),ans1(:,3),'gx');
i_fig=i_fig+1;

for i=1:rw
%     if ans1(i, 5)>0.985  % Set thresholds for corresponding points to remove wrong pairs
        line([ans1(i,2)+512,ans1(i,4)],[ans1(i,1),ans1(i,3)])
%     end
end
title('The resulting matching corners');

match_a1=[ans1(:,2),ans1(:,1),ones(rw,1)]';
match_a2=[ans1(:,4),ans1(:,3),ones(rw,1)]';

[hh, inliers] = ransacfithomography(match_a1, match_a2, rw, 10);
[height_wrap, width_wrap,~] = size(a1);
[height_unwrap, width_unwrap,~] = size(a2);

% USE INVERSE WARP METHOD
% DETERMINE THE SIZE OF THE WHOLE IMAGE
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

% DISPLAY IMAGE MOSIAC
figure(i_fig);
imshow(uint8(newImage));
 i_fig=i_fig+1;

%% harris corner detection
function [y1,y2,r,c]=harris(X)
% Corner detection, using the Harris algorithm
% The output is an image
% [result,cnt,r,c]=harris(X)

% f=rgb2gray(X);
f=X;

if(size(f,3)==3)
    ori_im = rgb2gray(uint8(f));  % Convert to grayscale image
end

%ori_im=double(f)/255;               % uint8 converted to 64-bit double precision double 64

fx = [5 0 -5;8 0 -8;5 0 -5];          % Gaussian function first order differential, x direction
%fx = [-2 -1 0 1 2];                 % Gradient operator in x direction
Ix = filter2(fx,ori_im);              % Filtering in the x direction
fy = [5 8 5;0 0 0;-5 -8 -5];          % Gaussian function first order differential, y direction
%fy = [-2;-1;0;1;2];                 % Gradient operator in y direction
Iy = filter2(fy,ori_im);              % Filtering in the y direction

Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix.*Iy;
clear Ix;
clear Iy;
 
h= fspecial('gaussian',[10 10 ],2);        % Generate a 7*7 Gaussian window function, sigma=2
 
Ix2 = filter2(h,Ix2);
Iy2 = filter2(h,Iy2);
Ixy = filter2(h,Ixy);                   % Gaussian filtering
 
height = size(ori_im,1);
width = size(ori_im,2);
result = zeros(height,width);           % Record the corner position, the value at the corner is 1, and the background is black
 
R = zeros(height,width);

for i = 1:height
    for j = 1:width
        M = [Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)];             % auto correlation matrix
        R(i,j) = det(M)-0.06*(trace(M))^2;   
    end
end

% Select the corner point according to the maximum value in the window
cnt = 0;
for i = 2:height-1
    for j = 2:width-1
        % Perform non-maximum suppression, window size 3*3
        if  R(i,j) > R(i-1,j-1) && R(i,j) > R(i-1,j) && R(i,j) > R(i-1,j+1) && R(i,j) > R(i,j-1) && R(i,j) > R(i,j+1) && R(i,j) > R(i+1,j-1) && R(i,j) > R(i+1,j) && R(i,j) > R(i+1,j+1)
            result(i,j) = 1;
            cnt = cnt+1;
        end
    end
end

Rsort=zeros(cnt,1);
[posr, posc] = find(result == 1);  % Return corner coordinates
for i=1:cnt
    Rsort(i)=R(posr(i),posc(i));
end

% Sort all corner R values and select the largest 100 as output
[Rsort,ix]=sort(Rsort,1);
Rsort=flipud(Rsort);
ix=flipud(ix);
ps=300;
posr2=zeros(ps,1);
posc2=zeros(ps,1);
for i=1:ps
    posr2(i)=posr(ix(i));
    posc2(i)=posc(ix(i));
end
 
y1=result;
y2=ps;
r=posr2;c=posc2;
return;
end
 
%% pairs match
function [res,ans_]=match(A1,cnt1,r1,c1,A2,cnt2,r2,c2)
% res=match(a1,a2)
% The best matching point in a2 will be found from a1, and the res extracted from a2 will be obtained, that is, one-way search

if(size(A1,3)==3)
    a1 = rgb2gray(uint8(A1));
end

if(size(A2,3)==3)
    a2 = rgb2gray(uint8(A2));
end
                         
a1=double(a1);
a2=double(a2);

% [m1,n1]=size(a1);
[m2,n2]=size(a2);
% res1=zeros(m1,n1);
res2=zeros(m2,n2);           % match point
ans_=zeros(100,5);
    
for s=1:cnt1          
    max=0; p=0;q=0;i=r1(s,1);j=c1(s,1);      % p.q store coordinates
    for v=1:cnt2
        m=r2(v,1);n=c2(v,1);
        
        mask_size=1;% 3*3 mask       
        if i+mask_size<340 && i-mask_size>0 && m+mask_size<340 && m-mask_size>0 && n+mask_size<340 && n-mask_size>0  && j+mask_size<340 && j-mask_size>0 
            
        else
            mask_size=1;
        end

        a1_norm = (sum(sum(a1(i-mask_size:i+mask_size,j-mask_size:j+mask_size).^2))).^0.5;
        a2_norm = (sum(sum(a2(m-mask_size:m+mask_size,n-mask_size:n+mask_size).^2))).^0.5;              
        ncc=sum(sum((a1(i-mask_size:i+mask_size,j-mask_size:j+mask_size)/a1_norm).*(a2(m-mask_size:m+mask_size,n-mask_size:n+mask_size)/a2_norm)));
      
        if ncc>max
             max=ncc;
             p=m;
             q=n;
        end
    end
    res2(p,q)=1;
    ans_(s,:)=[i,j,p,q,max];
 
end

res=res2;
return
end

%% remove same pairs
function newres = remove_same_pairs(res)
% res = [left_row, left_col, right_row, right_col, score]
total = 0;
for i=1:100
    if res(i, 5)~=0
        total = total+1;
    end
end

count = 0;
for s=1:total
    p = res(s,3); q = res(s,4);
    for i=1:total
        if i ~=s
            r = res(i,3); c = res(i,4);
            if p==r && q ==c
                if res(i, 5)>res(s,5)
                    if res(s, 5) ~= 0
                        res(s, 5) = 0;
                        count =count+1;
                    end
                else
                    if res(i, 5) ~= 0
                        res(i, 5) = 0;
                        count =count+1;
                    end
                end
            end
        end
    end
end

newres = zeros(total-count, 5);
k=1;
for i = 1:100
    if res(i, 5) ~= 0
        newres(k, :) = res(i, :);
        k = k+1;
    end
end
end

%% remove false pairs by slope and Euclidean distance
function newres = remove_false(res, imgsize)
% input
% res = [left_row, left_col, right_row, right_col, score]
% imgsize = img_height*img_width
[row, ~] = size(res);
dis = zeros(row,1);
slope = zeros(row,1);
for i = 1:row
    slope(i) = abs((res(i, 4)-res(2))/(res(i,3)-res(i,1)));
    dis(i) = sqrt((res(i,4)-res(i,2))^2+(res(i,3)-res(i,1))^2);
end


dis_threshold = imgsize*0.4;
slope_threshold = 6; %20бу
count=0;
for i = 1:row
    if dis(i)>dis_threshold || slope(i)<slope_threshold
        res(i,5) = 0;
        count =count+1;
    end
end

newres = zeros(row-count, 5);
k=1;
for i =1:row
    if res(i, 5) ~= 0
        newres(k,:) = res(i,:);
        k = k+1;
    end
end

end

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
function [hh, inliers] = ransacfithomography(ref_P, dst_P, npoints, threshold)
% 8-point RANSAC fitting
% Input:
% matcher_A - match points from image A, a matrix of 3xN, the third row is 1
% matcher_B - match points from image B, a matrix of 3xN, the third row is 1
% thd - distance threshold
% npoints - number of samples

ninlier = 0;
fpoints = 8; %number of fitting points
for i=1:2000
rd = randi([1 npoints],1,fpoints);
pR = ref_P(:,rd);
pD = dst_P(:,rd);
h = getHomographyMatrix(pR,pD,fpoints);
rref_P = h*ref_P;
rref_P(1,:) = rref_P(1,:)./rref_P(3,:);
rref_P(2,:) = rref_P(2,:)./rref_P(3,:);
error = (rref_P(1,:) - dst_P(1,:)).^2 + (rref_P(2,:) - dst_P(2,:)).^2;
n = nnz(error<threshold);
if(n >= npoints*.95)
hh=h;
inliers = find(error<threshold);
pause();
break;
elseif(n>ninlier)
ninlier = n;
hh=h;
inliers = find(error<threshold);
end 
end
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
mask = (newImage(:,:,1)>0 | newImage(:,:,2)>0 | newImage(:,:,3)>0);
mask = and(maskA, mask);

% GET THE OVERLAID REGION
[~,col] = find(mask);
left = min(col);
right = max(col);
mask = ones(size(mask));

if( x<2)
mask(:,left:right) = repmat(linspace(0,1,right-left+1),size(mask,1),1);
else
mask(:,left:right) = repmat(linspace(1,0,right-left+1),size(mask,1),1);
end

% BLEND EACH CHANNEL
warped_image(:,:,1) = warped_image(:,:,1).*mask;
warped_image(:,:,2) = warped_image(:,:,2).*mask;
warped_image(:,:,3) = warped_image(:,:,3).*mask;

% REVERSE THE ALPHA VALUE
if( x<2)
mask(:,left:right) = repmat(linspace(1,0,right-left+1),size(mask,1),1);
else
mask(:,left:right) = repmat(linspace(0,1,right-left+1),size(mask,1),1);
end
newImage(:,:,1) = newImage(:,:,1).*mask;
newImage(:,:,2) = newImage(:,:,2).*mask;
newImage(:,:,3) = newImage(:,:,3).*mask;

newImage(:,:,1) = warped_image(:,:,1) + newImage(:,:,1);
newImage(:,:,2) = warped_image(:,:,2) + newImage(:,:,2);
newImage(:,:,3) = warped_image(:,:,3) + newImage(:,:,3);
end
