clear all, close all, clc

folder='D:\FFSSAOOCT\2020.01.24\oct\2T\01_50nm_10000nms_100000fps_m0p18_C001H001S0001'
cd(folder)
% I=readmraw('example',[1 10]);

% filename='temp2';
% load(filename)
fid2=fopen('01_50nm_10000nms_100000fps_m0p18_C001H001S0001.mraw','r')

nWidth =   384
nHeight =   240
Frames=20501
Pixels = nWidth*nHeight;
I=zeros(Pixels,Frames,'uint16');

% CameraSetup.FrameRate=0;
for n=1:1:Frames
    I(:,n)=(fread(fid2,Pixels,'uint16'));
end
fclose(fid2);
N = [nWidth nHeight Frames];
ImageData.Images.RawImages=permute(reshape(I,N),[2 1 3]);

% cd 'D:\FFSSAOOCT\System characterization\New reference arm\Mirror\4'
for n=1:10
    figure(1)
    frame=ImageData.Images.RawImages(:,:,n);
imshow(frame,[0 3000]);
% imwrite(frame,strcat(sprintf('%0.6d',n),'.tif'))
pause(.02);
title(n)
end

% %% Compare with tif files
% 
% imagefiles = dir('*.tif');      
% nfiles = length(imagefiles);    % Number of files found
% 
% for ii=1:nfiles
%    currentfilename = imagefiles(ii).name;
%    currentimage = imread(currentfilename);
%    
%    figure(2)
%    subplot(1,3,1)
%    imshow(ImageData.Images.RawImages(:,:,ii),[]); 
%    title('mraw')
%    subplot(1,3,2)
%    imshow(currentimage,[]);
%    title('tif')
%    subplot(1,3,3)
%    
%    ratio=currentimage./ImageData.Images.RawImages(:,:,ii);
%    imshow(ratio,[])
%    title('mraw/tif')
%    
%    range=[min(ratio(:)) max(ratio(:))]
%    
%    pause(1);
% end
