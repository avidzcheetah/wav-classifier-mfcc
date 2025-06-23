% 2022e008, 2022e173
function cell2csv(filename, cellArray)
    fid = fopen(filename, 'w');
    for row = 1:size(cellArray, 1)
        fprintf(fid, '%s,%s\n', cellArray{row,1}, cellArray{row,2});
    end
    fclose(fid);
end
