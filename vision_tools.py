# ============================================================================
# 文件名: vision_tools.py
# 功能: 定义图像处理工具集，供 Agent 调用
# 说明: 提供图像读取、分析、转换、增强等功能
# ============================================================================

# ============================================================================
# 第1部分: 导入依赖库
# ============================================================================
from langchain_core.tools import tool
from datetime import datetime
import os
from typing import Optional
import base64

# 图像处理库
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️ 警告: Pillow 未安装，图像处理功能不可用")
    print("   安装命令: pip install Pillow")

# 计算机视觉库
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("⚠️ 警告: OpenCV 未安装，高级图像处理功能不可用")
    print("   安装命令: pip install opencv-python")


# ============================================================================
# 辅助函数
# ============================================================================

def _check_image_library():
    """检查图像处理库是否可用"""
    if not PILLOW_AVAILABLE:
        raise ImportError("Pillow 库未安装，请运行: pip install Pillow")


def _validate_image_path(image_path: str) -> str:
    """验证图片路径是否存在"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 检查是否为支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in supported_formats:
        raise ValueError(f"不支持的图片格式: {ext}，支持的格式: {supported_formats}")
    
    return image_path


def _get_image_info(image_path: str) -> dict:
    """获取图片基本信息"""
    _check_image_library()
    _validate_image_path(image_path)
    
    img = Image.open(image_path)
    info = {
        'format': img.format,
        'mode': img.mode,
        'size': img.size,
        'width': img.width,
        'height': img.height,
        'file_size': os.path.getsize(image_path),
        'file_path': image_path
    }
    img.close()
    
    return info


# ============================================================================
# 工具函数定义
# ============================================================================

# ----------------------------------------------------------------------------
# 工具1: get_image_info - 获取图片信息
# 功能: 读取图片的基本信息（尺寸、格式、大小等）
# 使用场景: 了解图片属性，决定后续处理方式
# ----------------------------------------------------------------------------
@tool
def get_image_info(image_path: str) -> str:
    """
    获取图片的详细信息
    
    Args:
        image_path: 图片文件路径
    
    Returns:
        str: 图片信息文本，包括格式、尺寸、文件大小等
    
    示例:
        get_image_info("photo.jpg")
    """
    try:
        info = _get_image_info(image_path)
        
        # 格式化文件大小
        file_size = info['file_size']
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"
        
        output = [
            "📊 图片信息:",
            "=" * 60,
            f"文件路径: {info['file_path']}",
            f"格式: {info['format']}",
            f"颜色模式: {info['mode']}",
            f"尺寸: {info['width']} x {info['height']} 像素",
            f"文件大小: {size_str}",
            "=" * 60
        ]
        
        return "\n".join(output)
    
    except Exception as e:
        return f"❌ 获取图片信息失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具2: resize_image - 调整图片尺寸
# 功能: 按比例或指定尺寸调整图片大小
# 使用场景: 压缩图片、适配显示需求
# ----------------------------------------------------------------------------
@tool
def resize_image(image_path: str, output_path: str, width: Optional[int] = None, 
                 height: Optional[int] = None, maintain_aspect_ratio: bool = True) -> str:
    """
    调整图片尺寸
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        width: 目标宽度（可选）
        height: 目标高度（可选）
        maintain_aspect_ratio: 是否保持宽高比（默认True）
    
    Returns:
        str: 操作结果
    
    示例:
        resize_image("input.jpg", "output.jpg", width=800)
        resize_image("input.jpg", "output.jpg", width=800, height=600, maintain_aspect_ratio=False)
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        original_size = img.size
        
        if maintain_aspect_ratio:
            # 保持宽高比
            if width and height:
                # 同时指定宽高，按较小的比例缩放
                ratio_w = width / img.width
                ratio_h = height / img.height
                ratio = min(ratio_w, ratio_h)
                new_size = (int(img.width * ratio), int(img.height * ratio))
            elif width:
                ratio = width / img.width
                new_size = (width, int(img.height * ratio))
            elif height:
                ratio = height / img.height
                new_size = (int(img.width * ratio), height)
            else:
                return "❌ 请至少指定宽度或高度"
        else:
            # 不保持宽高比
            if not width or not height:
                return "❌ 不保持宽高比时，必须同时指定宽度和高度"
            new_size = (width, height)
        
        # 调整尺寸
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 保存
        img_resized.save(output_path)
        img.close()
        img_resized.close()
        
        return f"✅ 图片尺寸调整成功\n原始尺寸: {original_size[0]}x{original_size[1]}\n新尺寸: {new_size[0]}x{new_size[1]}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 调整图片尺寸失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具3: convert_image_format - 转换图片格式
# 功能: 在不同图片格式之间转换（JPG, PNG, BMP, WebP等）
# 使用场景: 格式兼容性、压缩优化
# ----------------------------------------------------------------------------
@tool
def convert_image_format(image_path: str, output_path: str, quality: int = 95) -> str:
    """
    转换图片格式
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径（扩展名决定目标格式）
        quality: JPEG/WebP 质量 (1-100，默认95)
    
    Returns:
        str: 操作结果
    
    示例:
        convert_image_format("input.png", "output.jpg", quality=90)
        convert_image_format("input.jpg", "output.webp", quality=80)
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        
        # 确定输出格式
        output_ext = os.path.splitext(output_path)[1].lower()
        format_map = {
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.png': 'PNG',
            '.bmp': 'BMP',
            '.gif': 'GIF',
            '.tiff': 'TIFF',
            '.webp': 'WEBP'
        }
        
        if output_ext not in format_map:
            return f"❌ 不支持的输出格式: {output_ext}"
        
        output_format = format_map[output_ext]
        
        # 处理 RGBA 转 RGB（JPEG 不支持透明通道）
        if output_format == 'JPEG' and img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        # 保存
        save_kwargs = {}
        if output_format in ('JPEG', 'WEBP'):
            save_kwargs['quality'] = quality
        if output_format == 'PNG':
            save_kwargs['optimize'] = True
        
        img.save(output_path, format=output_format, **save_kwargs)
        img.close()
        
        original_size = os.path.getsize(image_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = (1 - output_size / original_size) * 100
        
        return f"✅ 格式转换成功\n原始格式: {os.path.splitext(image_path)[1]}\n目标格式: {output_ext}\n原始大小: {original_size / 1024:.2f} KB\n输出大小: {output_size / 1024:.2f} KB\n压缩率: {compression_ratio:.1f}%\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 格式转换失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具4: enhance_image - 增强图片质量
# 功能: 调整亮度、对比度、饱和度、锐度
# 使用场景: 改善图片视觉效果
# ----------------------------------------------------------------------------
@tool
def enhance_image(image_path: str, output_path: str, brightness: float = 1.0,
                  contrast: float = 1.0, saturation: float = 1.0, 
                  sharpness: float = 1.0) -> str:
    """
    增强图片质量
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        brightness: 亮度 (0.0-2.0，1.0为原图)
        contrast: 对比度 (0.0-2.0，1.0为原图)
        saturation: 饱和度 (0.0-2.0，1.0为原图)
        sharpness: 锐度 (0.0-2.0，1.0为原图)
    
    Returns:
        str: 操作结果
    
    示例:
        enhance_image("input.jpg", "output.jpg", brightness=1.2, contrast=1.1)
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        
        # 应用增强
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(sharpness)
        
        # 保存
        img.save(output_path)
        img.close()
        
        enhancements = []
        if brightness != 1.0:
            enhancements.append(f"亮度: {brightness}x")
        if contrast != 1.0:
            enhancements.append(f"对比度: {contrast}x")
        if saturation != 1.0:
            enhancements.append(f"饱和度: {saturation}x")
        if sharpness != 1.0:
            enhancements.append(f"锐度: {sharpness}x")
        
        return f"✅ 图片增强成功\n应用效果: {', '.join(enhancements) if enhancements else '无'}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 图片增强失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具5: apply_filter - 应用滤镜效果
# 功能: 应用常用滤镜（模糊、边缘检测、灰度等）
# 使用场景: 艺术效果、预处理
# ----------------------------------------------------------------------------
@tool
def apply_filter(image_path: str, output_path: str, filter_type: str = "blur", 
                 intensity: int = 2) -> str:
    """
    应用滤镜效果
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        filter_type: 滤镜类型 ("blur", "sharpen", "edge_enhance", "grayscale", "sepia")
        intensity: 强度 (1-5，仅对某些滤镜有效)
    
    Returns:
        str: 操作结果
    
    示例:
        apply_filter("input.jpg", "output.jpg", "blur", intensity=3)
        apply_filter("input.jpg", "output.jpg", "grayscale")
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        
        filter_type = filter_type.lower()
        
        if filter_type == "blur":
            # 高斯模糊
            for _ in range(intensity):
                img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        elif filter_type == "sharpen":
            # 锐化
            for _ in range(intensity):
                img = img.filter(ImageFilter.SHARPEN)
        
        elif filter_type == "edge_enhance":
            # 边缘增强
            for _ in range(intensity):
                img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        elif filter_type == "grayscale":
            # 灰度
            img = img.convert('L').convert('RGB')
        
        elif filter_type == "sepia":
            # 怀旧色
            img = img.convert('Sepia')
        
        else:
            return f"❌ 不支持的滤镜类型: {filter_type}。支持的类型: blur, sharpen, edge_enhance, grayscale, sepia"
        
        # 保存
        img.save(output_path)
        img.close()
        
        return f"✅ 滤镜应用成功\n滤镜类型: {filter_type}\n强度: {intensity}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 滤镜应用失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具6: crop_image - 裁剪图片
# 功能: 按指定区域裁剪图片
# 使用场景: 去除多余部分、提取感兴趣区域
# ----------------------------------------------------------------------------
@tool
def crop_image(image_path: str, output_path: str, left: int, top: int, 
               right: int, bottom: int) -> str:
    """
    裁剪图片
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        left: 左边界像素坐标
        top: 上边界像素坐标
        right: 右边界像素坐标
        bottom: 下边界像素坐标
    
    Returns:
        str: 操作结果
    
    示例:
        crop_image("input.jpg", "output.jpg", 100, 100, 500, 400)
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        
        # 验证裁剪区域
        if left >= right or top >= bottom:
            return "❌ 无效的裁剪区域：左边界必须小于右边界，上边界必须小于下边界"
        
        if right > img.width or bottom > img.height:
            return f"❌ 裁剪区域超出图片范围（图片尺寸: {img.width}x{img.height}）"
        
        # 裁剪
        img_cropped = img.crop((left, top, right, bottom))
        
        # 保存
        img_cropped.save(output_path)
        img.close()
        img_cropped.close()
        
        crop_width = right - left
        crop_height = bottom - top
        
        return f"✅ 图片裁剪成功\n裁剪区域: ({left}, {top}) 到 ({right}, {bottom})\n裁剪尺寸: {crop_width}x{crop_height}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 图片裁剪失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具7: rotate_image - 旋转图片
# 功能: 按角度旋转图片
# 使用场景: 校正方向、创意效果
# ----------------------------------------------------------------------------
@tool
def rotate_image(image_path: str, output_path: str, angle: float, 
                 expand: bool = True) -> str:
    """
    旋转图片
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        angle: 旋转角度（逆时针，度数）
        expand: 是否扩展画布以容纳整个图片（默认True）
    
    Returns:
        str: 操作结果
    
    示例:
        rotate_image("input.jpg", "output.jpg", 90)
        rotate_image("input.jpg", "output.jpg", -45, expand=False)
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        original_size = img.size
        
        # 旋转
        img_rotated = img.rotate(angle, expand=expand, resample=Image.Resampling.BICUBIC)
        
        # 保存
        img_rotated.save(output_path)
        img.close()
        img_rotated.close()
        
        new_size = img_rotated.size if hasattr(img_rotated, 'size') else original_size
        
        return f"✅ 图片旋转成功\n旋转角度: {angle}°\n原始尺寸: {original_size[0]}x{original_size[1]}\n新尺寸: {new_size[0]}x{new_size[1]}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 图片旋转失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具8: create_thumbnail - 创建缩略图
# 功能: 快速生成小尺寸缩略图
# 使用场景: 预览图、列表展示
# ----------------------------------------------------------------------------
@tool
def create_thumbnail(image_path: str, output_path: str, size: int = 128) -> str:
    """
    创建缩略图
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        size: 缩略图最大边长（像素，默认128）
    
    Returns:
        str: 操作结果
    
    示例:
        create_thumbnail("photo.jpg", "thumb.jpg", size=200)
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        img = Image.open(image_path)
        
        # 创建缩略图（保持宽高比）
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # 保存
        img.save(output_path)
        img.close()
        
        return f"✅ 缩略图创建成功\n最大边长: {size}px\n实际尺寸: {img.size[0]}x{img.size[1]}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 创建缩略图失败: {str(e)}"


# ----------------------------------------------------------------------------
# 工具9: add_watermark - 添加水印
# 功能: 在图片上添加文字或图片水印
# 使用场景: 版权保护、品牌标识
# ----------------------------------------------------------------------------
@tool
def add_watermark(image_path: str, output_path: str, text: str = "", 
                  watermark_image: str = "", position: str = "bottom-right",
                  opacity: float = 0.5) -> str:
    """
    添加水印
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        text: 水印文字（可选）
        watermark_image: 水印图片路径（可选）
        position: 位置 ("top-left", "top-right", "bottom-left", "bottom-right", "center")
        opacity: 透明度 (0.0-1.0，默认0.5)
    
    Returns:
        str: 操作结果
    
    示例:
        add_watermark("photo.jpg", "watermarked.jpg", text="© 2024", position="bottom-right")
    """
    try:
        _check_image_library()
        _validate_image_path(image_path)
        
        if not text and not watermark_image:
            return "❌ 请提供水印文字或水印图片"
        
        img = Image.open(image_path).convert('RGBA')
        
        # 创建透明图层
        txt_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
        
        if text:
            # 文字水印
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(txt_layer)
            
            # 尝试使用系统字体
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                try:
                    font = ImageFont.truetype("simhei.ttf", 40)  # 中文黑体
                except:
                    font = ImageFont.load_default()
            
            # 计算文字位置
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 根据位置计算坐标
            margin = 20
            if position == "top-left":
                pos = (margin, margin)
            elif position == "top-right":
                pos = (img.width - text_width - margin, margin)
            elif position == "bottom-left":
                pos = (margin, img.height - text_height - margin)
            elif position == "bottom-right":
                pos = (img.width - text_width - margin, img.height - text_height - margin)
            elif position == "center":
                pos = ((img.width - text_width) // 2, (img.height - text_height) // 2)
            else:
                pos = (img.width - text_width - margin, img.height - text_height - margin)
            
            # 绘制文字（半透明白色）
            alpha = int(255 * opacity)
            draw.text(pos, text, font=font, fill=(255, 255, 255, alpha))
        
        elif watermark_image:
            # 图片水印
            if not os.path.exists(watermark_image):
                return f"❌ 水印图片不存在: {watermark_image}"
            
            wm_img = Image.open(watermark_image).convert('RGBA')
            
            # 调整水印大小（不超过原图的1/4）
            max_wm_size = min(img.width, img.height) // 4
            if wm_img.width > max_wm_size or wm_img.height > max_wm_size:
                wm_img.thumbnail((max_wm_size, max_wm_size), Image.Resampling.LANCZOS)
            
            # 设置透明度
            if opacity < 1.0:
                alpha = wm_img.split()[3]
                alpha = alpha.point(lambda p: int(p * opacity))
                wm_img.putalpha(alpha)
            
            # 计算位置
            margin = 20
            if position == "top-left":
                pos = (margin, margin)
            elif position == "top-right":
                pos = (img.width - wm_img.width - margin, margin)
            elif position == "bottom-left":
                pos = (margin, img.height - wm_img.height - margin)
            elif position == "bottom-right":
                pos = (img.width - wm_img.width - margin, img.height - wm_img.height - margin)
            elif position == "center":
                pos = ((img.width - wm_img.width) // 2, (img.height - wm_img.height) // 2)
            else:
                pos = (img.width - wm_img.width - margin, img.height - wm_img.height - margin)
            
            # 粘贴水印
            txt_layer.paste(wm_img, pos, wm_img)
            wm_img.close()
        
        # 合并图层
        watermarked = Image.alpha_composite(img, txt_layer)
        
        # 转换为RGB并保存
        watermarked_rgb = watermarked.convert('RGB')
        watermarked_rgb.save(output_path)
        
        img.close()
        txt_layer.close()
        watermarked.close()
        watermarked_rgb.close()
        
        wm_type = "文字" if text else "图片"
        return f"✅ 水印添加成功\n水印类型: {wm_type}\n位置: {position}\n透明度: {opacity}\n保存至: {output_path}"
    
    except Exception as e:
        return f"❌ 添加水印失败: {str(e)}"


# ============================================================================
# 工具列表（用于导出）
# ============================================================================
__all__ = [
    'get_image_info',
    'resize_image',
    'convert_image_format',
    'enhance_image',
    'apply_filter',
    'crop_image',
    'rotate_image',
    'create_thumbnail',
    'add_watermark'
]
