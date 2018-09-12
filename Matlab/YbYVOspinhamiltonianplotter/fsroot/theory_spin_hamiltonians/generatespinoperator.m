function [sx, sy, sz] = generatespinoperator( spin )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
dim = 2*spin+1;
%initialize data
sx = zeros(dim);
sy = zeros(dim);
sz = zeros(dim);
m = (spin:-1:-spin);
if spin~=0
    %% create sx
    % stupid way
    % for row = 1:dim %m'
    %     for column = 1:dim %m
    %         switch column-row
    %             case 1
    %                 sx(row,column) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(column));
    %             case -1
    %                 sx(row,column) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(column));
    %         end
    %     end
    % end

    %not very elegant. could put it all in one loop, but then i'd need cases...
    % for row = 1
    %     sx(row, row+1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row+1));
    % end
    % for row = 2:dim-1
    %     sx(row, row+1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row+1));
    %     sx(row, row-1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row-1));
    % end
    % for row = dim
    %     sx(row, row-1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row-1));
    % end
    % %

    for row = 1:dim
        if row == 1
            sx(row, row+1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row+1));
        elseif row==dim
            sx(row, row-1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row-1));
        else
            sx(row, row+1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row+1));
            sx(row, row-1) = (1/2)*sqrt(spin*(spin+1)-m(row)*m(row-1));
        end
    end

    %% create sy
    %
    % for row = 1:dim %m'
    %     for column = 1:dim %m
    %         switch column-row
    %             case 1
    %                 sy(row,column) = -(1j/2)*sqrt(spin*(spin+1)-m(row)*m(column));
    %             case -1
    %                 sy(row,column) = (1j/2)*sqrt(spin*(spin+1)-m(row)*m(column));
    %         end
    %     end
    % end

    for row = 1:dim
        if row == 1
            sy(row, row+1) = -(1j/2)*sqrt(spin*(spin+1)-m(row)*m(row+1));
        elseif row == dim
            sy(row, row-1) = (1j/2)*sqrt(spin*(spin+1)-m(row)*m(row-1));
        else
            sy(row, row+1) = -(1j/2)*sqrt(spin*(spin+1)-m(row)*m(row+1));
            sy(row, row-1) = (1j/2)*sqrt(spin*(spin+1)-m(row)*m(row-1));
        end
    end

    %% create sz
    %
    for row = 1:dim %m'
        sz(row,row) = m(row);
    end
end
end

