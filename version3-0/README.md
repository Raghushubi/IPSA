Version 3.0: ocr and IC text experiments

In this version, I tried adding ocr to the existing cv pipeline to detect IC names from PCB images. Easyocr was integrated and instead of running ocr on all components, it is only applied on boxes that look like ICs (based on size and shape). I also added some basic image preprocessing to help ocr read text better. I also added an area based simple bucket classification(tiny, medium, large) for the components.

ocr was able to detect small labels like “101”,“C18”,etc, but proper IC names were mostly not detected. Even after trying 4K images, the IC text was still unclear in most cases because the text on chips is very small or blurry in normal pcb images.I also tried running ocr on the full image, but that mostly returned random short words or noise.

While doing this, higher resolution images caused too many components to be detected, so I had to adjust thresholds and merging logic to bring counts back close to Version 2. Right now ocr output is kept separate and the llm still uses only geometric component data. ocr will be added later once the text detection becomes more reliable.

Overall, Version 3.0 adds ocr support and IC candidate detection, and also shows that IC text extraction strongly depends on image quality.