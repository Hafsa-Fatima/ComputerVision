clear;
clc;
I=imread('Lena.png');
A=rgb2gray(I);
normImage = im2double(A);
B = normImage;
C = downSamplehalf(B);
D = downSamplehalf(C);


C_upsample = upSampletwice(C);
D_upsample = upSampletwice(upSampletwice(D));
figure,subplot(3,3,1),imshow(normImage);
title('Original image');
subplot(3,3,2),imshow(C_upsample);
title('upsample image once');
subplot(3,3,3),imshow(D_upsample);
title('upsample twice');

%problem 3
C_smooth = myGaussianSmoothing(C_upsample,11,1);
D_smooth = myGaussianSmoothing(upSampletwice(myGaussianSmoothing(upSampletwice(D),11,1)),11,1);

C_msmooth = medianFilter(C_upsample,3);
D_msmooth = medianFilter(upSampletwice(medianFilter(upSampletwice(D),3)),3);

subplot(3,3,4),imshow(normImage);
title('Original image');
subplot(3,3,5),imshow(C_smooth);
title('gaussian filter once on image');
subplot(3,3,6),imshow(D_smooth);
title('gaussian filter twice on image');
subplot(3,3,7),imshow(normImage);
title('Original image');
subplot(3,3,8),imshow(C_msmooth);
title('median filter once on image');
subplot(3,3,9),imshow(D_msmooth);
title('median filter twice on image');

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



