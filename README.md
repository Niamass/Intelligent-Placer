# Intelligent-Placer
 
### Описание программы 
По поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику определяется, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Предметы и горизонтальная поверхность, которые могут оказаться на фотографии, заранее известны. Также заранее известно направление вертикальной оси Z у этих предметов.

### Вход 
Фотография в формате jpg, на которой изображены:
  + предметы, помещенные на белом листе бумаги А4, лежащий на светлой горионтальной поверхности;
  + многоугольник, нарисованный темным маркером на белом листе бумаги А4, сфотографированной вместе с предметами.

### Выход 
Ответ в текстовом формате, записанный в файл answer_<имя входной фотографии>.txt:
  + "Yes" - предметы можно поместить в многоугольник;
  + "No" - предметы нельзя поместить в многоугольник.

### Требования к входным данным 
 #### Многоугольник
  + Толщина линии многоугольника не более 5 мм.
 #### Фон
  + Поверхность, на которой располагаются лист и предметы, гладкая, однородная;
  + Предметы располагаются на чистом белом листе формата А4, многоугольник нарисован на другом чистом белом листе формата А4;
  + Края листов с многоугольником и с предметами хорошо видны на поверхности;
  + Листы расположены на расстоянии не менее 1 см друг от друга и целиком находятся внутри фотографии.
 #### Предметы
  + Границы предметов четко выделяются на фоне белого листа бумаги;
  + Предметы не касаются или перекрывать друг друга;
  + Выбранные предметы целиком находятся внутри фотографии;
  + На фотографии отсутствуют предметы помимо тех, что представлены.
 #### Фотография
  + Фотография сделана сверху, камера при фотографировании удерживается параллельно поверхности с предметами;
  + Отсутствие на фотографии пересвеченных и серо-черных областей;
  + Отсутствие размытости фотографии: толщина линий границ предметов не более 10px;
  + Отсутствие сжатия и цветовой коррекции.
