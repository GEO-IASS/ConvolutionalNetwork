function image = VisualKernel(kernel)
edge_size = ceil(sqrt(size(kernel,3) * size(kernel,4)));

image = ones((size(kernel,1) + 1) * edge_size - 1,(size(kernel,2) + 1) * ...
    edge_size - 1);
for i = 1 : size(kernel,4)
    for j = 1 : size(kernel,3)
        num = (i - 1) * size(kernel,3) + j;
        x = floor((num - 1) / edge_size);
        y = mod(num - 1,edge_size);
        b = kernel(:,:,j,i);
        max_b = max(b(:));
        min_b = min(b(:));
        b = (b - min_b) ./ (max_b - min_b);
        image(x * (size(kernel,1) + 1) + 1 : x * (size(kernel,1) + 1) + size(kernel,1), ...
            y * (size(kernel,2) + 1) + 1 : y * (size(kernel,2) + 1) + size(kernel,2)) = b; 
    end
end
figure;
imshow(image);
end