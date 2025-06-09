import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_pdf_bboxes(pdf_path: str, page_num: int = 1, save_path: str = None):
    """
    可视化PDF中某一页的文本块和图像的包围盒（bounding box）。

    :param pdf_path: PDF 文件路径
    :param page_num: 要可视化的页码（从1开始）
    :param save_path: 如果提供，将保存图像到此路径，否则显示
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        page_image = page.to_image(resolution=150)
        fig, ax = plt.subplots(1, figsize=(10, 12))

        # 显示背景图像
        ax.imshow(page_image.original)

        # 提取文本块
        for block in page.extract_text_lines():
            x0, y0, x1, y1 = block["x0"], block["top"], block["x1"], block["bottom"]
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1.5, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(x0, y0 - 5, block['text'][:30], fontsize=6, color='green')

        # 提取图像块
        for img in page.images:
            x0, y0, x1, y1 = img['x0'], img['y0'], img['x1'], img['y1']
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x0, y0 - 5, "Image", fontsize=6, color='red')

        ax.set_title(f"Page {page_num} - BBox Visualization")
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved bbox visualization to {save_path}")
        else:
            plt.show()


# 示例调用方式
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize bounding boxes in a PDF page.")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument("--page", type=int, default=1, help="Page number to visualize (starting from 1)")
    parser.add_argument("--save", type=str, default=None, help="Path to save the output image")

    args = parser.parse_args()

    visualize_pdf_bboxes(args.pdf_path, page_num=args.page, save_path=args.save)