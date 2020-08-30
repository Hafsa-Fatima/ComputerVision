clear;
clc;
I=imread('Lena.png');
A=rgb2gray(I);
normImage = im2double(A);
noisy= gaussian_noise(normImage);
mnoisy= median_noise(normImage);
smooth= myGaussianSmoothing(noisy,3,1);
N = imnoise(normImage,'gaussian',0.1,0);
med= medianFilter(mnoisy,3);
figure,subplot(2,3,1),imshow(normImage);
title('Original');
subplot(2,3,2),imshow(noisy);
title('image with gaussian noise');
subplot(2,3,3),imshow(smooth);
title('gaussian filter');
subplot(2,3,4),imshow(normImage);
title('Original');
subplot(2,3,5),imshow(mnoisy);
title('image with noise');
subplot(2,3,6),imshow(med);
title('median filter');

function I_noisy=gaussian_noise(I)
N =normrnd(0,0.1,size(I));
I_noisy = I + N;
end


function I_noisy=median_noise(I)
N =normrnd(0,0.1,size(I));
for x= 1:size(I,1)
    for y = 1:size(I,1)

        if N(x,y)>0.2
            N(x,y)=1;
        else 
            N(x,y)=0;
        end
    end
end
    I_noisy = I + N;
end


function med= medianFilter(I,k)

    Output=zeros(size(I));

    I = padarray(I,[k-2 k-2]);
    s=size(Output,1);
    
    for i= 1:size(Output,1)
        for j= 1:size(Output,1)
            Output(i,j)= median(I(i:k+i-1,j:k+j-1),'all');
        end
    end
    
    med = Output;

end

function I_smooth =myGaussianSmoothing(I, k, s)
    
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
