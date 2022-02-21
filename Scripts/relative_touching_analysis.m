function [] = relative_touching_analysis(track_path,ref_path,save_path)
%relative touching analysis: computes relative antennal touches to the
%antennal, abdomenal, and body regions of partner bees. Relative touches
%are givens as touches per frame per edge fraction, so that a touch to a
%region with a small edge fraction (such as the antenna) is worth more than
%a touch to a larger region, roughly weighting touches by how unlikely they
%are to occur by chance.
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
%% Finding relative touches
%note that this step takes some time, ~100min on machine with 3.70GHz CPU, GeForce
%1080 Ti GPU
norm_touches = cell(pairs,1);
for trial = 1:colxcols+grpxgrps+isoxisos
    bee_1 = trial;%touching bee
    if ~iseven(trial)%touched bee
        bee_2 = trial + 1;
    else
        bee_2 = trial - 1;
    end
    track_1 = tracks{bee_1};
    track_2 = tracks{bee_2};
    norm_touching = zeros(length(track_1),3);
    parfor frame = 1:length(track_1)
        warning('off','MATLAB:polyshape:repairedBySimplify');%suppress warnings about irregular polygons
        %find antennal 'space', defined by body parts 2(head), 4&6
        %(proximal antennal joints), and 5&7(antennal tips) of touching bee
        bee_1_ant = polyshape(squeeze(track_1(frame,[5 4 2 6 7],:)));
        bee_1_ant = convhull(bee_1_ant);
        %find body 'space', defined by distal body parts (excluding wings)
        %5&7 (antennal tips),11&9 (foreleg tips), 15&13 (midleg tips),
        %and 17&19(hindleg tips) of touched bee
        bee_2_bod = polyshape(squeeze(track_2(frame,[5 7 11 15 19 17 13 9],:)));
        bee_2_bod = convhull(bee_2_bod);
        %reject any frames missing points necessary to define a
        %normalizeable body edge
        if length(bee_2_bod.Vertices) < 8
            if sum(ismember(bee_2_bod.Vertices,ones(size(bee_2_bod.Vertices)).*[track_2(frame,5,1) track_2(frame,5,2)],'rows')) ~= 1 ...
                    || sum(ismember(bee_2_bod.Vertices,ones(size(bee_2_bod.Vertices)).*[track_2(frame,7,1) track_2(frame,7,2)],'rows')) ~= 1 ...
                    || sum(ismember(bee_2_bod.Vertices,ones(size(bee_2_bod.Vertices)).*[track_2(frame,19,1) track_2(frame,19,2)],'rows')) ~= 1 ...
                    || sum(ismember(bee_2_bod.Vertices,ones(size(bee_2_bod.Vertices)).*[track_2(frame,17,1) track_2(frame,17,2)],'rows')) ~= 1
                continue
            end
        end
        %determine if there are any interesection of the edges of the
        %antennal space of the touching bee and the body space of the
        %touched bee
        %code adapted from:
        %U. Murat Erdem (2021). Fast Line Segment Intersection 
        %(https://www.mathworks.com/matlabcentral/fileexchange/27205-fast-line-segment-intersection),
        %MATLAB Central File Exchange. Retrieved September 16, 2021. 
        XY1 = bee_1_ant.Vertices;
        XY1 = [XY1 circshift(XY1,1,1)];
        XY2 = bee_2_bod.Vertices;
        XY2 = [XY2 circshift(XY2,1,1)];
        [n_rows_1,n_cols_1] = size(XY1);
        [n_rows_2,n_cols_2] = size(XY2);
        X1 = repmat(XY1(:,1),1,n_rows_2);
        X2 = repmat(XY1(:,3),1,n_rows_2);
        Y1 = repmat(XY1(:,2),1,n_rows_2);
        Y2 = repmat(XY1(:,4),1,n_rows_2);
        XY2 = XY2';
        X3 = repmat(XY2(1,:),n_rows_1,1);
        X4 = repmat(XY2(3,:),n_rows_1,1);
        Y3 = repmat(XY2(2,:),n_rows_1,1);
        Y4 = repmat(XY2(4,:),n_rows_1,1);
        X4_X3 = (X4-X3);
        Y1_Y3 = (Y1-Y3);
        Y4_Y3 = (Y4-Y3);
        X1_X3 = (X1-X3);
        X2_X1 = (X2-X1);
        Y2_Y1 = (Y2-Y1);
        numerator_a = X4_X3 .* Y1_Y3 - Y4_Y3 .* X1_X3;
        numerator_b = X2_X1 .* Y1_Y3 - Y2_Y1 .* X1_X3;
        denominator = Y4_Y3 .* X2_X1 - X4_X3 .* Y2_Y1;
        u_a = numerator_a ./ denominator;
        u_b = numerator_b ./ denominator;
        INT_B = (u_a >= 0) & (u_a <= 1) & (u_b >= 0) & (u_b <= 1);
        if sum(sum(INT_B)) == 0
            continue 
        end
        %find indices known points ended up in the polyshape object
        lant_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,5,:)),2)==2);
        rant_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,7,:)),2)==2);
        rfleg_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,11,:)),2)==2);
        rmleg_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,15,:)),2)==2);
        rhleg_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,19,:)),2)==2);
        lhleg_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,17,:)),2)==2);
        lmleg_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,13,:)),2)==2);
        lfleg_vert_ind = find(sum(ismember(bee_2_bod.Vertices,track_2(frame,9,:)),2)==2);
        %find total edge and ant/abdo/body(depending on which points are available) edges
        tot_len = perimeter(bee_2_bod);
        ant_len = pdist2(bee_2_bod.Vertices(lant_vert_ind,:),bee_2_bod.Vertices(rant_vert_ind,:));
        abdo_len = pdist2(bee_2_bod.Vertices(lhleg_vert_ind,:),bee_2_bod.Vertices(rhleg_vert_ind,:));
        if isempty(lfleg_vert_ind)
            if isempty(lmleg_vert_ind)
                lbod_len = pdist2(bee_2_bod.Vertices(lant_vert_ind,:),bee_2_bod.Vertices(lhleg_vert_ind,:));
            else
                lbod_len = pdist2(bee_2_bod.Vertices(lant_vert_ind,:),bee_2_bod.Vertices(lmleg_vert_ind,:)) + ...
                    pdist2(bee_2_bod.Vertices(lmleg_vert_ind,:),bee_2_bod.Vertices(lhleg_vert_ind,:));
            end
        elseif isempty(lmleg_vert_ind)
            lbod_len = pdist2(bee_2_bod.Vertices(lant_vert_ind,:),bee_2_bod.Vertices(lfleg_vert_ind,:)) + ...
                    pdist2(bee_2_bod.Vertices(lfleg_vert_ind,:),bee_2_bod.Vertices(lhleg_vert_ind,:));
        else
            lbod_len = pdist2(bee_2_bod.Vertices(lant_vert_ind,:),bee_2_bod.Vertices(lfleg_vert_ind,:)) + ...
                pdist2(bee_2_bod.Vertices(lfleg_vert_ind,:),bee_2_bod.Vertices(lmleg_vert_ind,:)) + ...
                pdist2(bee_2_bod.Vertices(lmleg_vert_ind,:),bee_2_bod.Vertices(lhleg_vert_ind,:));
        end
        if isempty(rfleg_vert_ind)
            if isempty(rmleg_vert_ind)
                rbod_len = pdist2(bee_2_bod.Vertices(rant_vert_ind,:),bee_2_bod.Vertices(rhleg_vert_ind,:));
            else
                rbod_len = pdist2(bee_2_bod.Vertices(rant_vert_ind,:),bee_2_bod.Vertices(rmleg_vert_ind,:)) + ...
                    pdist2(bee_2_bod.Vertices(rmleg_vert_ind,:),bee_2_bod.Vertices(rhleg_vert_ind,:));
            end
        elseif isempty(rmleg_vert_ind)
            rbod_len = pdist2(bee_2_bod.Vertices(rant_vert_ind,:),bee_2_bod.Vertices(rfleg_vert_ind,:)) + ...
                    pdist2(bee_2_bod.Vertices(rfleg_vert_ind,:),bee_2_bod.Vertices(rhleg_vert_ind,:));    
        else
            rbod_len = pdist2(bee_2_bod.Vertices(rant_vert_ind,:),bee_2_bod.Vertices(rfleg_vert_ind,:)) + ...
                pdist2(bee_2_bod.Vertices(rfleg_vert_ind,:),bee_2_bod.Vertices(rmleg_vert_ind,:)) + ...
                pdist2(bee_2_bod.Vertices(rmleg_vert_ind,:),bee_2_bod.Vertices(rhleg_vert_ind,:));
        end
        bod_len= lbod_len + rbod_len;
        int_segs = sum(INT_B,1) ~= 0;
        XY2 = XY2.';
        touch_keeper = zeros(1,3);
        %find relative touches, that is, each touch is scored as
        %total available edge/touched edge, so that score increases as
        %targetted edge shrinks
        for crossing = 1:length(int_segs)
            if int_segs(crossing) == 1
                if sum(ismember(XY2(crossing,:),track_2(frame,5,:))) == 2 && ...
                        sum(ismember(XY2(crossing,:),track_2(frame,7,:))) == 2
                    touch_keeper(1,1) = tot_len/ant_len;
                elseif sum(ismember(XY2(crossing,:),track_2(frame,17,:))) == 2 && ...
                        sum(ismember(XY2(crossing,:),track_2(frame,19,:))) == 2
                    touch_keeper(1,2) = tot_len/abdo_len;
                else
                    touch_keeper(1,3) = tot_len/bod_len;
                end
            end
        end
        norm_touching(frame,:) = touch_keeper;
    end
    norm_touches{trial} = norm_touching;
end
%% Visualizing results
trial_nums = [colxcols grpxgrps isoxisos];
reltouch_keeper = zeros(3,sum(trial_nums));
for touch_cat = 1:3
    for trial = 1:sum(trial_nums)
        int_data = norm_touches{trial}(:,touch_cat);
        int_data(isinf(int_data)) = [];
        reltouch_keeper(touch_cat,trial) = mean(int_data);
    end
end
%for all homogeneous pairs
data = mean(reltouch_keeper,2);
err = std(reltouch_keeper,[],2)/sqrt(length(reltouch_keeper));
hold off
b = bar(1:3,data,'FaceColor','flat','EdgeColor','none');
b.CData(1,:) = [1 .8398 0];
b.CData(2,:) = [.17 .75 .17];
b.CData(3,:) = [1 0 0];
hold on
er = errorbar(1:3,data,err,err,'CapSize',20);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
er.LineWidth = 3;
set(gca,'linewidth',4)
set(gca,'fontsize',20)
set(gca,'box','off')
set(gcf,'position',[383 0 1086 911])
xticks(.75:2.75)
xticklabels({'ant-ant','ant-abdo','ant-body'})
xtickangle(45)
set(gca, 'TickLength',[0 0])
ylabel('touches per frame per edge fraction')
saveas(gcf,strcat(save_path,'rel_touches_homogeneous','.png'),'png');
%for all colxcol pairs
data = mean(reltouch_keeper(:,find(grouper==1)),2);
err = std(reltouch_keeper(:,find(grouper==1)),[],2)/sqrt(length(reltouch_keeper(:,find(grouper==1))));
hold off
b = bar(1:3,data,'FaceColor','flat','EdgeColor','none');
b.CData(1,:) = [1 .8398 0];
b.CData(2,:) = [.17 .75 .17];
b.CData(3,:) = [1 0 0];
hold on
er = errorbar(1:3,data,err,err,'CapSize',20);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
er.LineWidth = 3;
set(gca,'linewidth',4)
set(gca,'fontsize',20)
set(gca,'box','off')
% set(gcf,'position',[383 0 1086 911])
ylim([0 .34])
xticks(.75:2.75)
xticklabels({'ant-ant','ant-abdo','ant-body'})
xtickangle(45)
set(gca, 'TickLength',[0 0])
ylabel('touches per frame per edge fraction')
saveas(gcf,strcat(save_path,'rel_touches_colxcol','.png'),'png');
%for all grpxgrp pairs
data = mean(reltouch_keeper(:,find(grouper==2)),2);
err = std(reltouch_keeper(:,find(grouper==2)),[],2)/sqrt(length(reltouch_keeper(:,find(grouper==2))));
hold off
b = bar(1:3,data,'FaceColor','flat','EdgeColor','none');
b.CData(1,:) = [1 .8398 0];
b.CData(2,:) = [.17 .75 .17];
b.CData(3,:) = [1 0 0];
hold on
er = errorbar(1:3,data,err,err,'CapSize',20);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
er.LineWidth = 3;
set(gca,'linewidth',4)
set(gca,'fontsize',20)
set(gca,'box','off')
% set(gcf,'position',[383 0 1086 911])
ylim([0 .34])
xticks(.75:2.75)
xticklabels({'ant-ant','ant-abdo','ant-body'})
xtickangle(45)
set(gca, 'TickLength',[0 0])
ylabel('touches per frame per edge fraction')
saveas(gcf,strcat(save_path,'rel_touches_grpxgrp','.png'),'png');
%for all isoxiso pairs
data = mean(reltouch_keeper(:,find(grouper==3)),2);
err = std(reltouch_keeper(:,find(grouper==3)),[],2)/sqrt(length(reltouch_keeper(:,find(grouper==3))));
hold off
b = bar(1:3,data,'FaceColor','flat','EdgeColor','none');
b.CData(1,:) = [1 .8398 0];
b.CData(2,:) = [.17 .75 .17];
b.CData(3,:) = [1 0 0];
hold on
er = errorbar(1:3,data,err,err,'CapSize',20);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';
er.LineWidth = 3;
set(gca,'linewidth',4)
set(gca,'fontsize',20)
set(gca,'box','off')
% set(gcf,'position',[383 0 1086 911])
ylim([0 .34])
xticks(.75:2.75)
xticklabels({'ant-ant','ant-abdo','ant-body'})
xtickangle(45)
set(gca, 'TickLength',[0 0])
ylabel('touches per frame per edge fraction')
saveas(gcf,strcat(save_path,'rel_touches_isoxiso','.png'),'png');
%% Checking for significance
fprintf('for all homogeneous pairings \n');
fprintf('Wilcoxon rank sum ant-ant vs. ant-abdo p = %.4g\n', ranksum(reltouch_keeper(1,:),reltouch_keeper(2,:)));
fprintf('Wilcoxon rank sum ant-ant vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(1,:),reltouch_keeper(3,:)));
fprintf('Wilcoxon rank sum ant-abdo vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(2,:),reltouch_keeper(3,:)));
fprintf('\n')
fprintf('for all colxcol pairings \n');
fprintf('Wilcoxon rank sum ant-ant vs. ant-abdo p = %.4g\n', ranksum(reltouch_keeper(1,find(grouper==1)),reltouch_keeper(2,find(grouper==1))));
fprintf('Wilcoxon rank sum ant-ant vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(1,find(grouper==1)),reltouch_keeper(3,find(grouper==1))));
fprintf('Wilcoxon rank sum ant-abdo vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(2,find(grouper==1)),reltouch_keeper(3,find(grouper==1))));
fprintf('\n')
fprintf('for all grpxgrp pairings \n');
fprintf('Wilcoxon rank sum ant-ant vs. ant-abdo p = %.4g\n', ranksum(reltouch_keeper(1,find(grouper==2)),reltouch_keeper(2,find(grouper==2))));
fprintf('Wilcoxon rank sum ant-ant vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(1,find(grouper==2)),reltouch_keeper(3,find(grouper==2))));
fprintf('Wilcoxon rank sum ant-abdo vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(2,find(grouper==2)),reltouch_keeper(3,find(grouper==2))));
fprintf('\n')
fprintf('for all isoxiso pairings \n');
fprintf('Wilcoxon rank sum ant-ant vs. ant-abdo p = %.4g\n', ranksum(reltouch_keeper(1,find(grouper==3)),reltouch_keeper(2,find(grouper==3))));
fprintf('Wilcoxon rank sum ant-ant vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(1,find(grouper==3)),reltouch_keeper(3,find(grouper==3))));
fprintf('Wilcoxon rank sum ant-abdo vs. ant-body p = %.4g\n', ranksum(reltouch_keeper(2,find(grouper==3)),reltouch_keeper(3,find(grouper==3))));
fprintf('\n')
end