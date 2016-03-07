function image = Visual4DDatum(data)
edge_size = ceil(sqrt(size(data,4)));
image = ones((size(data,1) + 1) * edge_size - 1,(size(data,2) + 1) * edge_size - 1,size(data,3));
for i = 1 : size(data,4)
    x = floor((i - 1) / edge_size);
    y = mod(i - 1,edge_size);
    b = data(:,:,:,i);
    min_b = min(b(:));
    max_b = max(b(:));
    b = (b - min_b) ./ (max_b - min_b);
    image(x * (size(data,1) + 1) + 1 : x * (size(data,1) + 1) + size(data,1), ...
        y * (size(data,2) + 1) + 1 : y * (size(data,2) + 1) + size(data,2),:) ...
        = b;
end
figure;
imshow(image);
end