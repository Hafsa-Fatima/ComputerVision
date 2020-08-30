clear;
clc;
I=imread('Lena.png');
A=rgb2gray(I);
normImage = im2double(A);
figure,subplot(2,3,1),imshow(normImage);
title('Original image');
B = normImage;
C = downSamplehalf(B);
D = downSamplehalf(C);

subplot(2,3,2),imshow(C);
title('Downsample 1/2 ');
subplot(2,3,3),imshow(D);
title('Downsample 1/4');
C_upsample = upSampletwice(C);
D_upsample = upSampletwice(upSampletwice(D));
subplot(2,3,4),imshow(normImage);
title('Original image');
subplot(2,3,5),imshow(C_upsample);
title('Upsample Once image');
subplot(2,3,6),imshow(D_upsample);
title('Upsample twice image');


function downsample = downSamplehalf(X)
    downsample = X(1:2:end,1:2:end);
end

function upsample = upSampletwice(X)
 s=size(X);
 newsize=2*s(1,1);
 Z = zeros(newsize);
 Z(1:2:end,1:2:end)= X;
 upsample = Z;
end