# Настройка GPU для ускорения обучения

## Нужен ли мне GPU?

**НЕТ** - для выполнения задания GPU не обязателен. CPU версия работает везде.

**ДА** - если хотите ускорить обучение в 2-3 раза.

---

## Поддерживаемые платформы

### ✅ Windows + NVIDIA GPU (GeForce, RTX, Quadro)
**Ускорение: 2-3x**

### ✅ Mac M1/M2/M3 (Apple Silicon)
**Ускорение: 1.5-2x**

### ❌ Windows + AMD GPU
**Не рекомендуется** - плохая поддержка в PyTorch

### ❌ Intel GPU
**Не поддерживается**

---

## Автоматическое определение

Все скрипты проекта **автоматически** определяют доступное устройство:

```python
if torch.cuda.is_available():
    device = torch.device('cuda')      # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device('mps')       # Apple Metal
else:
    device = torch.device('cpu')       # CPU
```

**Вам ничего не нужно менять в коде!**

---

## Установка: Windows + NVIDIA GPU

### Вариант 1: Только PyTorch с CUDA (рекомендуется)

```cmd
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

PyTorch включает все необходимые CUDA библиотеки.

### Вариант 2: С установкой CUDA Toolkit (если нужен)

1. Скачайте CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Установите (~3 GB, ~10 минут)
3. Установите PyTorch:
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Проверка GPU:

```cmd
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Нет')"
```

Должно показать:
```
CUDA: True
GPU: NVIDIA GeForce RTX ...
```

---

## Установка: Mac M1/M2/M3

**Ничего дополнительно не нужно!**

Стандартная установка PyTorch уже поддерживает Apple Metal (MPS):

```bash
pip install -r requirements.txt
```

Проверка:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

---

## Частые проблемы

### "CUDA: False" - GPU не определяется

**Причины:**

1. **Устаревшие драйверы NVIDIA**
   - Скачайте последние с nvidia.com или через GeForce Experience
   - CUDA 12.1 требует драйверы 530+
   - Перезагрузите после установки

2. **Несовместимость версии CUDA**
   - Если драйверы старые, используйте CUDA 11.8:
     ```cmd
     pip uninstall torch torchvision
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
     ```

3. **Проверьте версию драйвера:**
   ```cmd
   nvidia-smi
   ```
   Должна показать информацию о GPU и версию драйвера.

4. **Нет NVIDIA GPU**
   - Проверьте в диспетчере устройств (Win + X → Диспетчер устройств → Видеоадаптеры)
   - Если там AMD/Intel - используйте CPU

### "MPS: False" на Mac

- MPS доступен только на Apple Silicon (M1/M2/M3)
- Intel Mac не поддерживают MPS - используйте CPU

### Обучение вылетает с ошибкой на GPU

**Решение:**
1. Добавьте в начало скрипта (только для отладки):
   ```python
   import os
   os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # для Mac
   ```

2. Или переключитесь на CPU:
   ```python
   device = torch.device('cpu')
   ```

---

## Сравнение времени обучения

### Baseline модель (20 эпох)

| Устройство | Время |
|------------|-------|
| CPU (Intel i5/i7) | 5-10 мин |
| GPU (NVIDIA RTX 3060) | 2-3 мин |
| Apple M1/M2 (MPS) | 3-5 мин |

### Полные эксперименты (24 × 20 эпох)

| Устройство | Время |
|------------|-------|
| CPU | ~40 мин |
| GPU (NVIDIA) | ~15 мин |
| Apple M1/M2 (MPS) | ~20 мин |

---

## Рекомендации

1. **Для новичков:** Начните с CPU версии
   - Проще установить
   - Работает везде
   - Для учебного проекта скорости достаточно

2. **Если есть NVIDIA GPU:** Используйте CUDA
   - Значительное ускорение
   - Стабильная поддержка

3. **На Mac M1/M2/M3:** MPS работает "из коробки"
   - Умеренное ускорение
   - Иногда могут быть проблемы с некоторыми операциями

4. **AMD GPU на Windows:** Используйте CPU
   - ROCm плохо работает на Windows
   - Не стоит тратить время на настройку

---

## FAQ

**Q: Обязательно ли использовать GPU для сдачи задания?**
A: Нет, CPU версия полностью подходит.

**Q: Почему обучение на GPU не намного быстрее?**
A: MLP - относительно небольшая модель, оверхед на копирование данных на GPU заметен. На CNN ускорение было бы больше.

**Q: Можно ли использовать Google Colab?**
A: Да, но придется загружать файлы в Colab. Для локального запуска проще.

**Q: Что лучше - CUDA 11.8 или 12.1?**
A: Если драйверы свежие (530+) - используйте 12.1. Если старые - 11.8.

**Q: Нужен ли CUDA Toolkit?**
A: Обычно нет - PyTorch включает все необходимое. Нужен только если компилируете расширения.

---

## Итого

- ✅ **CPU версия работает везде** - рекомендуется для начинающих
- ✅ **GPU ускоряет в 2-3 раза** - полезно при большом числе экспериментов
- ✅ **Скрипты автоматически определяют GPU** - ничего менять не нужно
- ✅ **Для задания GPU не обязателен** - CPU достаточно

Выбирайте то, что удобнее для вас!
