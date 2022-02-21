function [] = antant_touching_analysis(track_path,ref_path,save_path)
%antant touching analysis: computes the fraction of time pairs of bees are
%in antennae-to-antennae contract as defined by the overlap of their
%antennal 'spaces', the shaped defined by the head, antennal joints, and
%antennal tips of each bee.
%
%track_path: folder with analysis h5s
%ref_path: location of refpoints.mat
%save_path: folder where output figures are saved
%create save_path folder if needed
if ~exist(save_path,'dir')
    mkdir(save_path)
end
%% Compiling tracks
%create array of tracking files in the folder
tracking_listing = dir(strcat(track_path,'\*.analysis.h5'));
tracking_listing = strtrim(string(char(tracking_listing.name)));
%load in reference points to center coordinates from each trial to same
%point
load(ref_path);
%set up some indexing variables for homogeneous pairs
colxcols = 2*sum(contains(tracking_listing,"colxcol"));
grpxgrps = 2*sum(contains(tracking_listing,"grpxgrp"));
isoxisos = 2*sum(contains(tracking_listing,"isoxiso"));
tracks = cell(colxcols+grpxgrps+isoxisos,1);
grouper = zeros(colxcols+grpxgrps+isoxisos,1);
type_names = ["colxcol","grpxgrp","isoxiso"];
bee_idx = 1;
for type = 1:3
    indices = find(contains(tracking_listing,type_names(type)));
    for tracking_id = indices.'
        raw_tracks = h5read(strcat(track_path,"\",tracking_listing(tracking_id)),'/tracks');
        %fill in any missing values and and upsample any mismatched frame rates
        warning('off','MATLAB:interp1:NaNstrip');%suppress warnings about NaNs during interpolation
        if contains(tracking_listing(tracking_id),"fr")
            time_points = (179942/size(raw_tracks,1))*(0:size(raw_tracks,1)-1) + 1;%179942 being the average number of frames in the 100fps videos
            raw_tracks = interp1(time_points,raw_tracks,1:179942,'pchip');
        else
            raw_tracks = interp1(1:size(raw_tracks,1),raw_tracks,1:size(raw_tracks,1),'pchip');
        end
        for bee = 1:2
            tracks_mat = raw_tracks(:,:,:,bee);
            tracks_mat(:,:,1) = tracks_mat(:,:,1) + cp_shuffle(tracking_id,1);
            tracks_mat(:,:,2) = tracks_mat(:,:,2) + cp_shuffle(tracking_id,2);
            tracks{bee_idx} = tracks_mat;
            grouper(bee_idx) = type;
            bee_idx = bee_idx + 1;
        end
    end
end
%% Finding ant-ant touches
%note that this step takes some time, ~45min on machine with 3.70GHz CPU, GeForce
%1080 Ti GPU
%determine ant-ant touching by overlap of antennal regions of both bees
%(defined as the polygon created by the head and antennal coordinates of
%the bee)
antant_touches = cell(colxcols+grpxgrps+isoxisos,1);
for trial = 1:2:colxcols+grpxgrps+isoxisos
    track_1 = tracks{trial};
    track_2 = tracks{trial+1};
    touching = zeros(length(track_1),1);
    parfor frame = 1:length(track_1)
        warning('off','MATLAB:polyshape:repairedBySimplify');%suppress warnings about irregular polygons
        %find antennal 'spaces', defined by body parts 2(head), 4&6
        %(proximal antennal joints), and 5&7(antennal tips)
        bee_1_ant = polyshape(squeeze(track_1(frame,[5 4 2 6 7],:)));
        bee_1_ant = convhull(bee_1_ant);
        bee_2_ant = polyshape(squeeze(track_2(frame,[5 4 2 6 7],:)));
        bee_2_ant = convhull(bee_2_ant);
        %times when atennal spaces are overlapping will be defined as
        %touches
        if overlaps(bee_1_ant,bee_2_ant)
            touching(frame) = 1;
        end
    end
    antant_touches{trial} = touching;
end
%% Visualizing results
trial_nums = [colxcols grpxgrps isoxisos];
touchfrac_data = [];
for trial = 1:2:colxcols+grpxgrps+isoxisos
    touchfrac_data(trial) = mean(antant_touches{trial});
end
hold off
boxplot(touchfrac_data(1:2:end),grouper(1:2:end),'symbol','k');
a = get(get(gca,'children'),'children');   % Get the handles of all the objects
t = get(a,'tag');   % List the names of all the objects 
box1 = a(4:9);   % The 7th object is the first box
set(box1, 'Color', 'k','linewidth',1.5)
hold on
swarmchart(ones(colxcols/2,1),touchfrac_data(1:2:max(find(grouper==1))),[],[242/255 204/255 143/255],'filled','XJitterWidth',0.4)
swarmchart(2*ones(grpxgrps/2,1),touchfrac_data(max(find(grouper==1))+1:2:max(find(grouper==2))),[],[206/255 83/255 116/255],'filled','XJitterWidth',0.4)
swarmchart(3*ones(isoxisos/2,1),touchfrac_data(max(find(grouper==2))+1:2:max(find(grouper==3))),[],[61/255 64/255 91/255],'filled','XJitterWidth',0.4)
set(gca,'linewidth',4)
set(gca,'fontsize',20)
set(gca,'box','off')
set(gcf,'position',[183 61 600 911])
xticks([1 2 3])
xticklabels({'Col','Grp','Iso'})
set(gca, 'TickLength',[0 0])
ylabel('fraction time ant-ant')
%% Checking for significance
fprintf('Wilcoxon rank sum colxcol vs. grpxgrp p = %.4g\n', ranksum(touchfrac_data(1:2:max(find(grouper==1))).',touchfrac_data(max(find(grouper==1))+1:2:max(find(grouper==2))).'));
fprintf('Wilcoxon rank sum colxcol vs. isoxiso p = %.4g\n', ranksum(touchfrac_data(1:2:max(find(grouper==1))).',touchfrac_data(max(find(grouper==2))+1:2:max(find(grouper==3))).'));
fprintf('Wilcoxon rank sum grpxgrp vs. isoxiso p = %.4g\n', ranksum(touchfrac_data(max(find(grouper==1))+1:2:max(find(grouper==2))).',touchfrac_data(max(find(grouper==2))+1:2:max(find(grouper==3))).'));
fprintf('\n')
fprintf('FKtest for homogeneity of variances colxcol vs. grpxgrp p = %.4g\n', Fapprox_FK(touchfrac_data(1:2:max(find(grouper==1))).',touchfrac_data(max(find(grouper==1))+1:2:max(find(grouper==2))).'));
fprintf('FKtest for homogeneity of variances colxcol vs. isoxiso p = %.4g\n', Fapprox_FK(touchfrac_data(1:2:max(find(grouper==1))).',touchfrac_data(max(find(grouper==2))+1:2:max(find(grouper==3))).'));
fprintf('FKtest for homogeneity of variances grpxgrp vs. isoxiso p = %.4g\n', Fapprox_FK(touchfrac_data(max(find(grouper==1))+1:2:max(find(grouper==2))).',touchfrac_data(max(find(grouper==2))+1:2:max(find(grouper==3))).'));
fprintf('\n')
end
%%
function p = Fapprox_FK(sample_1, sample_2)
%FK test of homogeneity of variance between two samples
%built with reference to
%Trujillo-Ortiz, A., R. Hernandez-Walls and N. Castro-Castro. (2009).FKtest:
%Fligner-Killeen test for homogeneity of variances. A MATLAB file. [WWW document].
%URL http://www.mathworks.com/matlabcentral/fileexchange/25040
%and also with reference to 
%Conover, W. J., Johnson, M. E. and Johnson, M. M. (1981), A Comparative
%Study of Tests for Homogeneity of Variances, with Applications to the
%Outer Continental Shelf Bidding Data. Technometrics, 23(4):351-361.
n1 = length(sample_1);
n2 = length(sample_2);
%rank all |xi - x~|
ranks = tiedrank([abs(sample_1 - median(sample_1)) ; abs(sample_2 - median(sample_2))]);
%calculate FK scores
a = norminv(0.5 + ranks/(2*(n1 + n2 +1)));
%calculate chi squared statistic
chisquared = (n1*(mean(a(1:n1))-mean(a))^2 + n2*(mean(a(1+n1:end))-mean(a))^2)/var(a);
%calculate f statistic
F = chisquared/((n1 + n2 -1 - chisquared)/(n1 + n2 -2));
p = 1 - fcdf(F,1,n1+n2-2);
end











