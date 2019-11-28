using System;
using System.Linq;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.Generic;
using CGLabPlatform;


// Объявляем производный класс с любым именем от GFXApplicationTemplate<T>. Его реализация 
// находиться в сборке CGLabPlatform, следовательно надо добавить в Reference ссылку на  
// эту сборку (также нужны ссылки на сборки: Sysytem, System.Drawing и System.Windows.Forms) 
// и объявить, что используется простанство имен - using CGLabPlatform. И да верно, класс 
// является  абстрактным - дело в колдунстве, что происходит внутри базового класса - по факту
// он динамически объявляет новый класс производный от нашего класса и создает прокси объект.

public abstract class CGLab01 : GFXApplicationTemplate<CGLab01>
{
    // Точка входа приложения - просто передаем управление базовому классу
    // вызывая метод RunApplication(), дабы он сделал всю оставшуюся работу
    // Впринципе, одной этой строчки, в объявленном выше классе, вполне 
    // достаточно чтобы приложение можно было уже запустить
    [STAThread] static void Main() { RunApplication(); }

    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // --- Часть 1: Объявление и работа со свойствами
    // ---------------------------------------------------------------------------------

    // Для добавления свойств и связанных с ними элементов управления на панели свойств
    // приложения нужно пометить открытое (public) свойство одним из следующих аттрибутов:
    //  * DisplayNumericProperty - численное значение, типа int, float, double и т.д.
    //  * DisplayCheckerProperty - переключатель для значения типа bool
    //  * DisplayTextBoxProperty - текстовое значение типа string
    //  * DisplayEnumListProperty - меню выбора для значения типа перечисления (Enum)
    //  У всех аттрибутов первый параметр задает начальное значение, затем значение 
    //  типа string - отображаемое название. Для численных значений дополнительно
    //  можно задать величину шага (инкремент) и граничные значение (мин, макс)
    //  Сами свойства можно определить несколькими способами:

    // 1 - Краткая форма: Просто объявляем свойства как абстрактное или виртуальное
    [DisplayNumericProperty(0, 1, "Сдвиг по X")]
    public abstract int ShiftX { get; set; }

    [DisplayNumericProperty(0, 0.1, "Сдвиг по Y", -1000)]
    public virtual double ShiftY { get; set; }

    // Все остальные формы объявления имеют лишь смысл, когда нужно добавить некий  код, 
    // выполняемый непосредственно при изменении или получении того или иного значения.

    // 2 - Полная форма: Делаем почти все самостоятельно
    [DisplayCheckerProperty(true, "Включить вращения")]
    public bool EnableRot {         // Cвойства на деле синтаксический сахар после компиляции
        get {                       // посути получаем методы: getSomething() и setSomething()
            return _EnableRot;      // Поэтому туда и можно пихать код и поэтому же требуется
        }                           // дополнительное поле для хранения связанного значения.
        set {
            _EnableRot = value;
            // ... - какой-то код, который надо выполнить при изменении значения
            base.OnPropertyChanged();   // Реализация привязки свойства в обе стороны, без
        }                               // этого изменение данного свойства не будет приво-
    }                                   // дить к обновлению содержимого элемента управления
    private bool _EnableRot;


    // 3 - Упрощенная форма: Объявляем свойство как виртуальное, base.OnPropertyChanged() 
    //                       будет добавленно за нас, за счет чего код сократится
    [DisplayTextBoxProperty("Hellow World", "Заголовок")] 
    public virtual string LabelTxt {
        get { return _LabelTxt; }
        set { _LabelTxt = value;
            // ... - какой-то код, который надо выполнить при изменении значения
        }
    }
    private string _LabelTxt;


    // 4 - Еще более упрощенный вариант: Дополнительно к предыдущему избавляемся от
    //     объявления поля, вместо этого значение будет храниться в колекции базового
    //     типа, а доступ к нему осуществляться через методы Get<T> и Set<T>. Конечно
    //     при таком подходе скорость доступа существенно падает, но если не обращаться
    //     к нему в циклак во время отрисовки это не критично.
    [DisplayEnumListProperty(DrawingMode.DIB, "Метод рисования")]
    public DrawingMode DrawMode {
        get { return Get<DrawingMode>(); }
        set { if (!Set<DrawingMode>(value)) return;
            // ... - какой-то код, который надо выполнить при изменении значения
        }
    }

    // Выше было использованно свойство для собственного перечесления (enum), как не
    // трудно догадаться это его объявление. Аттрибутами Description - задается выводимый
    // текст, когда лениво их можно опустить.
    public enum DrawingMode {
        [Description("Graphics")] GFX,
        [Description("Surface")]  DIB
    }


    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // --- Часть 2: Инициализация данных, управления и поведения приложения
    // ---------------------------------------------------------------------------------


    // Если нужна какая-то инициализация данных при запуске приложения, можно реализовать ее
    // в перегрузке данного события, вызываемого единожды перед отображением окна приложения
    protected override void OnMainWindowLoad(object sender, EventArgs args)
    {
        // Созданное приложение имеет два основных элемента управления:
        // base.RenderDevice - левая часть экрана для рисования
        // base.ValueStorage - правая панель для отображения и редактирования свойств

        // Пример изменения внешниго вида элементов управления (необязательный код)
        base.RenderDevice.BufferBackCol = 0xB0;
        base.ValueStorage.Font = new Font("Arial", 12f);
        base.ValueStorage.ForeColor = Color.Firebrick;
        base.ValueStorage.RowHeight = 30;
        base.ValueStorage.BackColor = Color.BlanchedAlmond;
        base.MainWindow.BackColor = Color.DarkGoldenrod;
        base.ValueStorage.RightColWidth = 50;
        base.VSPanelWidth = 300;
        base.VSPanelLeft = true;
        base.MainWindow.Size = new Size(960, 640);
        
        
        // Реализация управления мышкой с зажатыми левой и правой кнопкой мыши
        base.RenderDevice.MouseMoveWithRightBtnDown += (s, e) => {
            ShiftX += e.MovDeltaX;
            ShiftY -= e.MovDeltaY;
        };
        base.RenderDevice.MouseMoveWithLeftBtnDown += (s, e) => {
            ShiftX += 10 * e.MovDeltaX;
            ShiftY -= 10 * e.MovDeltaY;
        };

        // Реализация управления клавиатурой
        RenderDevice.HotkeyRegister(Keys.Up,    (s, e) => ++ShiftY);
        RenderDevice.HotkeyRegister(Keys.Down,  (s, e) => --ShiftY);
        RenderDevice.HotkeyRegister(Keys.Left,  (s, e) => --ShiftX);
        RenderDevice.HotkeyRegister(Keys.Right, (s, e) => ++ShiftX);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Up,    (s, e) => ShiftY +=10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Down,  (s, e) => ShiftY -=10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Left,  (s, e) => ShiftX -=10);
        RenderDevice.HotkeyRegister(KeyMod.Shift, Keys.Right, (s, e) => ShiftX +=10);

        // ... расчет каких-то параметров или инициализация ещё чего-то, если нужно
    }


    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // --- Часть 3: Формирование изображение и его отрисовка на экране
    // ---------------------------------------------------------------------------------


    // При надобности добавляем нужные поля, методы, классы и тд и тп.
    private double angle = 0;



    // Перегружаем главный метод. По назначению он анологичен методу OnPaint() и предназначен
    // для формирования изображения. Однако в отличии от оного он выполняется паралелльно в
    // другом потоке и вызывается непрерывно. О текущей частоте вызовов можно судить по 
    // счетчику числа кадров в заголовке окна (конечно в режиме отладки скорость падает).
    // Помимо прочего он обеспечивает более высокую скорость рисования и не мерцает.
    protected override void OnDeviceUpdate(object s, GDIDeviceUpdateArgs e)
    {
        double sx = ShiftX; // С точки зрения производительности часто используемые свойства лучше 
        double sy = ShiftY; // сохранить в локальные переменные или обращаться к связаным с ними полям
        double lx1 = sx + e.Width/2, ly1 = sy + e.Heigh/2, lx2 = e.Width/4, ly2 = e.Heigh/4;
        double[] txy = new double[] { 0, -60, -40, +20, +50, +40 };

        if (EnableRot)
            angle += 0.1 * e.Delta;
        var sinf = Math.Sin(angle * Math.PI/180);
        var cosf = Math.Cos(angle * Math.PI/180);
        for (int i = 0; i < 3; ++i) {
            var x = txy[2*i];
            var y = txy[2*i +1];
            txy[2*i]       = x * cosf - y * sinf + e.Width/2 + sx;
            txy[2 * i + 1] = x * sinf + y * cosf + e.Heigh/2 - sy;
        }

        if (DrawMode == DrawingMode.GFX) {
            // Стандартный способ отображения при помощи штатных средств
            // + Больше гибкость, например можно задать толщину линий
            // + Для треугольников есть сглаживание
            for (double fi = 0; fi < 2*Math.PI; fi += Math.PI/32)
                e.Graphics.DrawLine(Pens.GhostWhite, lx1, ly1,
                    +sx + lx1 + lx2 * Math.Cos(fi) - ly2 * Math.Sin(fi),
                    -sy + ly1 + lx2 * Math.Sin(fi) + ly2 * Math.Cos(fi));
            e.Graphics.FillPolygon(Brushes.Chocolate, new PointF[] {
                new PointF((float)txy[0], (float)txy[1]),
                new PointF((float)txy[2], (float)txy[3]),
                new PointF((float)txy[4], (float)txy[5])
            });

            e.Graphics.FillRectangle(Brushes.DodgerBlue, 9, 35, 200, 12);
        } else {
            // Собственная реализация отображения (CGLabPlatform.Drawing)
            // + Выше скорость работы (у меня разница более чем в 5 раз)
            // + Лучшее качество сглаженных прямых
            // + Градиентная закраска треугольника (т.е. задание различных 
            //   цветов для каждой из верши), что будет востребаванно
            for (double fi = 0; fi < 2 * Math.PI; fi += Math.PI/32)
                e.Surface.DrawLine(Color.GhostWhite.ToArgb(), lx1, ly1,
                    +sx + lx1 + lx2 * Math.Cos(fi) - ly2 * Math.Sin(fi),
                    -sy + ly1 + lx2 * Math.Sin(fi) + ly2 * Math.Cos(fi));
            e.Surface.DrawTriangle(Color.Chocolate.ToArgb(), txy[0], txy[1], txy[2], txy[3], txy[4], txy[5]);

            e.Surface.FillRectangle(Color.DodgerBlue.ToArgb(), 10, 35, 200, 12);
        }

        // Вторая реализация отображения осуществляется путем прямого доступа
        // к поверхности связанной с объектом Graphics. Поэтому возможно
        // одновременное использование обоих методов работы с изображением.
        // Так вне зависимости от выбранного метода, используется объект
        // Graphics для вывода текста.
        e.Graphics.DrawString(LabelTxt, new Font("Arial", 15f), Brushes.Chartreuse, 10f, 10f );
    }



    // В случае, если в потоке приложения потребуется выполнить некий код, 
    // с синхронизацией с потоком рендера, то реализуем это следующим образом
    //    lock (RenderDevice.LockObj) {
    //        // ... какой-то код
    //    }


}
