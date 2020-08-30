clear;
clc;
I=imread('Lena.png');
A=rgb2gray(I);
normImage = im2double(A);
figure,title("when s is constant ie s=1")
subplot(1,6,1),imshow(normImage);
title("original")

k =[3, 5, 7, 11, 51];
s = [0.1, 1, 2, 3, 5];

for i = 1:5
    I_smooth =myGaussianSmoothing(normImage,k(i), 1);
    subplot(1,6,i+1),imshow(I_smooth);
    p=strcat('When k=',num2str(k(i)));
    title(p)
end

figure,title("when k is constant ie k=11")
subplot(1,6,1),imshow(normImage);
title("original")
subplot(1,6,2),imshow(normImage);
title("When s=0.1")

for i = 2:5
    I_smooth =myGaussianSmoothing(normImage,11, s(i));
    subplot(1,6,i+1),imshow(I_smooth);
    p=strcat('When s=',num2str(s(i)));
    title(p)
end

%I_smooth =myGaussianSmoothing(normImage,11, 0.1);


function I_smooth =myGaussianSmoothing(I, k, s)
   % H = fspecial('gaussian',k,s);
  %  I_smooth = imfilter(I,H,'replicate');
    
    [x,y]=meshgrid(-k:k,-k:k);

    X = size(x,1)-1;
    Y = size(y,1)-1;
    e = -(x.^2+y.^2)/(2*s*s);
    kerFilter= exp(e)/(2*pi*s*s);

    Output=zeros(size(I));

    I = padarray(I,[k k]);

    for i = 1:size(I,1)-X
        for j =1:size(I,2)-Y
            Temp = I(i:i+X,j:j+X).*kerFilter;
            Output(i,j)=sum(Temp(:));
        end
    end

    I_smooth = Output;
end
