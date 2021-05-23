import torch
from model import SSD300

def main():
    model = SSD300(n_classes=4)
    model.load_state_dict(torch.load('saved/model_best.pth'))
    model.eval()

    input_tensor = torch.rand(1, 3, 300, 300)
    mobile = torch.jit.trace(model, input_tensor)
    mobile.save('mobile.pt')

if __name__ == '__main__':
    main()