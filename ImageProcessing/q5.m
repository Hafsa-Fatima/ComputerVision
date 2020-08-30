clear;
clc;
I=imread('Lena.png');
A=rgb2gray(I);
normImage = im2double(A);

[mag, ori] = mySobelFilter(normImage);
hsv = ori;
hsv(:,:,2)=mag;
hsv(:,:,3)=mag;

figure,subplot(1,5,1),imshow(normImage);
title('Original');
subplot(1,5,2),imshow(mag);
title('Magnitude');
subplot(1,5,3),imshow(ori);
title('Orientation');
subplot(1,5,4),imshow(hsv);
title('HSV');
subplot(1,5,5),imshow(hsv2rgb(hsv));
title('Edge Visualization');


function [mag, ori] = mySobelFilter(I)
mag=zeros(size(I));
ori=zeros(size(I));
for i=1:size(I,1)-2
    for j=1:size(I,2)-2
        X = ((2*I(i+2,j+1)+I(i+2,j)+I(i+2,j+2))-(2*I(i,j+1)+I(i,j)+I(i,j+2)));
        Y = ((2*I(i+1,j+2)+I(i,j+2)+I(i+2,j+2))-(2*I(i+1,j)+I(i,j)+I(i+2,j)));
        Gx(i,j)=X;
        mag(i,j)=sqrt(X.^2+Y.^2);
        ori(i,j)=atan((Y./X));
    end
end
end
