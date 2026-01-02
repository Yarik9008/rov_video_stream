#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт для вывода списка доступных камер и их разрешений"""

import cv2


def get_resolutions(cap):
    """Получает поддерживаемые разрешения камеры."""
    resolutions = set()
    test_res = [(640, 480), (800, 600), (1024, 768), (1280, 720), (1280, 960), 
                (1600, 1200), (1920, 1080), (2560, 1440), (3840, 2160)]
    
    # Текущее разрешение
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolutions.add((w, h))
    
    # Проверяем стандартные разрешения
    for width, height in test_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        resolutions.add((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    return sorted(resolutions)


def main():
    """Главная функция."""
    cameras = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened() or not cap.read()[0]:
            cap.release()
            continue
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolutions = get_resolutions(cap)
        cap.release()
        
        cameras.append({
            'index': i,
            'current': (w, h),
            'resolutions': resolutions
        })
    
    if not cameras:
        print("Камеры не найдены")
        return
    
    print(f"Найдено камер: {len(cameras)}\n")
    for cam in cameras:
        res_str = ", ".join([f"{w}x{h}" for w, h in cam['resolutions']])
        print(f"Камера #{cam['index']}: {res_str}")


if __name__ == "__main__":
    main()
