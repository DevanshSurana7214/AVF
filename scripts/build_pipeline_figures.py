"""Publication vector schematics for the frozen ROI AVF implementation.

No image encoders, experts or new methods are implied. Only aggregate counts
are used; the feature tiles and probability glyphs are explicitly schematic.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon, Circle, PolyLine
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE
OUT = HERE/'paper/figures'
COUNTS = ROOT/'results/class_pattern_counts.csv'
# Arial reproduces the archived layout. ReportLab's Vera is a portable fallback.
import reportlab
font_dir = Path(reportlab.__file__).resolve().parent/'fonts'
regular = Path('C:/Windows/Fonts/arial.ttf')
bold = Path('C:/Windows/Fonts/arialbd.ttf')
pdfmetrics.registerFont(TTFont('Arial', str(regular if regular.exists() else font_dir/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(bold if bold.exists() else font_dir/'VeraBd.ttf')))
INK='#263442'; MUTED='#5B6774'; GRID='#CFD7DF'; PALE='#F5F7FA'
MODS=['MRI','FDG','Amyloid','Tau']
COLORS=['#267E88','#C38C3C','#7964A3','#B56578']
TREE='#9B713E'; NEURAL='#536E99'; TRAIN='#A45E3D'
CLASSES=['CN','MCI','Dementia']; CLASSCOL=['#4E79A7','#D29C43','#B15D65']

def light(color, amount=.9):
    c=HexColor(color)
    return Color(c.red+(1-c.red)*amount,c.green+(1-c.green)*amount,c.blue+(1-c.blue)*amount)

class Figure:
    def __init__(self,w,h):
        self.w,self.h=w,h
        self.d=Drawing(w,h)
        self.d.add(Rect(0,0,w,h,fillColor=white,strokeColor=None))
    def rect(self,x,y,w,h,fill=white,stroke=GRID,lw=.8,r=0,dash=None):
        a=Rect(x,self.h-y-h,w,h,rx=r,ry=r,fillColor=HexColor(fill) if isinstance(fill,str) else fill,
               strokeColor=(HexColor(stroke) if isinstance(stroke,str) else stroke) if stroke else None,strokeWidth=lw)
        if dash:a.strokeDashArray=dash
        self.d.add(a)
    def line(self,x1,y1,x2,y2,color=INK,lw=1,dash=None):
        a=Line(x1,self.h-y1,x2,self.h-y2,strokeColor=HexColor(color),strokeWidth=lw)
        if dash:a.strokeDashArray=dash
        self.d.add(a)
    def poly(self,points,color=INK,lw=1,dash=None):
        a=PolyLine([v for x,y in points for v in (x,self.h-y)],strokeColor=HexColor(color),strokeWidth=lw,fillColor=None)
        if dash:a.strokeDashArray=dash
        self.d.add(a)
    def arrow(self,x1,y1,x2,y2,color=INK,lw=1.2,dash=None,head=5):
        self.line(x1,y1,x2,y2,color,lw,dash)
        angle=math.atan2(y2-y1,x2-x1)
        pts=[(x2,y2),(x2-head*math.cos(angle)+head*.45*math.sin(angle),y2-head*math.sin(angle)-head*.45*math.cos(angle)),
             (x2-head*math.cos(angle)-head*.45*math.sin(angle),y2-head*math.sin(angle)+head*.45*math.cos(angle))]
        self.d.add(Polygon([v for x,y in pts for v in (x,self.h-y)],fillColor=HexColor(color),strokeColor=None))
    def text(self,x,y,s,size=13,color=INK,bold=False,anchor='start',width=None):
        font='Arial-Bold' if bold else 'Arial'
        if width is not None:
            assert pdfmetrics.stringWidth(s,font,size)<=width, (s,width)
        self.d.add(String(x,self.h-y,s,fontName=font,fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
    def circle(self,x,y,r,fill=white,stroke=INK,lw=1):
        self.d.add(Circle(x,self.h-y,r,fillColor=HexColor(fill) if isinstance(fill,str) else fill,strokeColor=HexColor(stroke),strokeWidth=lw))
    def panel(self,x,y,w,h,label,title,accent=INK):
        self.rect(x,y,w,h,fill=white,stroke=GRID,r=2)
        self.rect(x,y,w,29,fill=light(accent,.93),stroke=None)
        self.text(x+12,y+20,f'({label})  {title}',13.5,bold=True)
    def tile(self,x,y,color,present=True,w=56,h=26,cols=6,rows=3):
        # Schematic feature matrix, never a subject image or measured value.
        for row in range(rows):
            for col in range(cols):
                fill=light(color,.35+.11*((row*3+col)%5)) if present else HexColor('#ECEFF2')
                self.rect(x+col*w/cols,y+row*h/rows,w/cols-.8,h/rows-.8,fill=fill,stroke=None)
        self.rect(x,y,w,h,fill=None,stroke=color if present else '#A7B0BA',lw=.7)
        if not present:
            self.line(x,y+h,x+w,y,color='#ADB5BF',lw=.9)
    def bits(self,x,y,bits,cell=22,labels=False):
        for i,b in enumerate(bits):
            self.rect(x+i*cell,y,cell-2,20,fill=light(COLORS[i],.84) if b=='1' else '#F0F2F4',stroke=COLORS[i] if b=='1' else GRID)
            self.text(x+i*cell+(cell-2)/2,y+14,b,12.5,bold=True,anchor='middle',color=COLORS[i] if b=='1' else MUTED)
            if labels:self.text(x+i*cell+(cell-2)/2,y+34,['M','F','A','T'][i],11,anchor='middle',color=MUTED)
    def module(self,x,y,w,h,title,lines=(),accent=NEURAL,fill=None):
        self.rect(x,y,w,h,fill=fill if fill is not None else light(accent,.95),stroke=accent,r=2)
        self.text(x+w/2,y+18,title,13,bold=True,anchor='middle',width=w-10)
        for i,s in enumerate(lines):self.text(x+w/2,y+35+i*15,s,12,anchor='middle',color=MUTED,width=w-10)
    def probability(self,x,y,w=97,h=35):
        self.line(x,y+h,x+w,y+h,color=GRID,lw=.8)
        for i,v in enumerate([.23,.57,.20]):
            xx=x+10+i*29
            self.rect(xx,y+h-v*h*1.35,15,v*h*1.35,fill=CLASSCOL[i],stroke=None)
            self.text(xx+7.5,y+h+15,['CN','MCI','D'][i],10.5,anchor='middle',color=MUTED)
    def tree(self,x,y):
        self.line(x+13,y,x,y+17,TREE); self.line(x+13,y,x+26,y+17,TREE)
        for xx in [0,26]:
            self.line(x+xx,y+17,x+xx-7,y+31,TREE,.8);self.line(x+xx,y+17,x+xx+7,y+31,TREE,.8)
        for xx,yy in [(13,0),(0,17),(26,17),(-7,31),(7,31),(19,31),(33,31)]:
            self.rect(x+xx-2.3,y+yy-2.3,4.6,4.6,fill=light(TREE,.5),stroke=TREE,lw=.7)
    def save(self,name):
        OUT.mkdir(exist_ok=True,parents=True)
        renderPDF.drawToFile(self.d,str(OUT/f'{name}.pdf'))
        renderSVG.drawToFile(self.d,str(OUT/f'{name}.svg'))
        doc=pdfium.PdfDocument(str(OUT/f'{name}.pdf'))
        doc[0].render(scale=3).to_pil().save(OUT/f'{name}.png')

def overview():
    f=Figure(900,316)
    f.panel(10,10,186,254,'a','Regional inputs',COLORS[0])
    f.panel(221,10,191,254,'b','Feature preparation',NEURAL)
    f.panel(437,10,229,254,'c','Complementary learners',TREE)
    f.panel(691,10,199,254,'d','Probability fusion',NEURAL)
    for i,(mod,color,bit) in enumerate(zip(MODS,COLORS,'1010')):
        yy=52+35*i
        f.text(23,yy+16,mod,12,bold=True,color=color)
        f.tile(96,yy,color,bit=='1',w=75,h=27)
        if bit=='0':f.text(133,yy+18,'absent',10,color=MUTED,anchor='middle')
    f.text(23,216,'Example availability mask',11.5,color=MUTED)
    f.bits(46,225,'1010',24)
    for yy,title,sub in [(54,'Retain reliable features','≤20% missing in training'),(102,'Standardize and clip','Training-fold mean and SD'),(150,'Impute and flag','Zero fill + missing indicators'),(198,'Keep modality identity','Append four availability bits')]:
        f.module(234,yy,164,40,title,[sub],accent=NEURAL)
        if yy<198:f.arrow(316,yy+40,316,yy+48,head=4,lw=.9)
    f.arrow(197,149,220,149,head=5)
    f.arrow(413,149,436,149,head=5)
    f.module(451,54,201,84,'LightGBM',[ 'Same regional features + mask'],accent=TREE)
    for xx in [476,530,584]:f.tree(xx,101)
    f.module(451,146,201,91,'Masked neural network',['Four modality MLP encoders','Masked pooling + joint loss'],accent=NEURAL)
    for i,col in enumerate(COLORS):
        f.rect(468+i*33,211,23,12,fill=light(col,.5),stroke=col,lw=.7)
    f.text(607,221,'64-D',10.5,color=MUTED)
    f.poly([(443,149),(443,96),(451,96)],lw=1)
    f.poly([(443,149),(443,189),(451,189)],lw=1)
    f.arrow(652,96,702,96,color=TREE)
    f.arrow(652,189,702,189,color=NEURAL)
    f.text(709,73,'Average three seeds',12,bold=True)
    f.text(725,103,'p̄T',15,color=TREE,bold=True)
    f.text(725,195,'p̄N',15,color=NEURAL,bold=True)
    f.poly([(750,96),(766,96),(766,150)],color=TREE)
    f.poly([(750,189),(766,189),(766,165)],color=NEURAL)
    f.circle(766,158,14,fill='#F7F8FA',stroke=NEURAL)
    f.text(766,164,'+',19,anchor='middle',bold=True)
    f.text(785,118,'α',14,color=TREE)
    f.text(783,189,'1 − α',12,color=NEURAL)
    f.arrow(781,158,856,158,color=NEURAL)
    f.text(820,149,'pF',14,color=NEURAL,bold=True)
    f.probability(777,196,w=96,h=29)
    f.text(788,254,'Schematic probabilities',10.5,anchor='middle',color=MUTED)
    f.rect(10,278,656,27,fill=light(TRAIN,.95),stroke=TRAIN,dash=[4,3])
    f.text(21,296,'Training only: remove one modality (p = 0.5) + expected-count joint weights → neural branch',12,color=TRAIN)
    f.poly([(553,278),(553,256),(553,237)],color=TRAIN,dash=[4,3])
    f.text(698,287,'Selected τ = 0: no prior correction',11,color=MUTED)
    f.text(698,303,'Developmental selection; not fully nested',10.7,color=MUTED)
    f.save('roi_pipeline')

def architecture():
    f=Figure(900,320)
    f.panel(10,9,880,84,'a','Training-fold preparation of each regional feature block',COLORS[0])
    steps=[('Filter columns','≤20% missing'),('Standardize','Mean / population SD'),('Clip and zero fill','Range [−5, 5]'),('Expose missingness','Append feature flags')]
    for i,(title,sub) in enumerate(steps):
        x=25+i*217
        f.module(x,46,195,37,title,[sub],accent=COLORS[0])
        if i<3:f.arrow(x+195,69,x+216,69,head=5)
    f.panel(10,105,481,188,'b','Modality-specific encoders',NEURAL)
    f.panel(506,105,384,188,'c','Masked pooling and diagnosis head',NEURAL)
    for i,(mod,col,bit) in enumerate(zip(MODS,COLORS,'1010')):
        y=144+i*34
        f.text(22,y+18,mod,12,bold=True,color=col)
        f.tile(89,y,col,bit=='1',w=44,h=25)
        f.arrow(134,y+12,155,y+12,color=col if bit=='1' else MUTED,dash=None if bit=='1' else [3,2])
        f.module(157,y-1,126,28,'MLP encoder',accent=col,fill=light(col,.91) if bit=='1' else HexColor('#F0F2F4'))
        f.arrow(284,y+12,307,y+12,color=col)
        for j in range(6):f.rect(310+j*10,y+4,8,17,fill=light(col,.4) if bit=='1' else '#E4E8ED',stroke=col if bit=='1' else GRID,lw=.5)
        f.text(380,y+17,f'h{i+1}',12,color=col)
        f.arrow(398,y+12,421,y+12,color=col)
        f.rect(424,y,53,25,fill=light(col,.91) if bit=='1' else '#F0F2F4',stroke=col if bit=='1' else GRID,r=2)
        f.text(450,y+17,f'a{i+1}' if bit=='1' else '0',13,anchor='middle',bold=True,color=col if bit=='1' else MUTED)
        f.arrow(478,y+12,579,208,color=col if bit=='1' else '#A7B0BA',lw=1.1 if bit=='1' else .7,dash=None if bit=='1' else [3,3])
    f.text(330,137,'64-D tokens',11,color=MUTED)
    f.text(421,137,'Gate + mask',10.5,color=MUTED)
    f.circle(593,208,16,fill='#F7F9FC',stroke=NEURAL)
    f.text(593,214,'Σ',18,anchor='middle',bold=True,color=NEURAL)
    f.text(530,176,'a = softmax(masked scores)',11,color=MUTED)
    f.text(526,245,'h: weighted 64-D token',11,color=NEURAL)
    f.arrow(610,208,650,208,color=NEURAL)
    f.circle(663,208,10,fill=white,stroke=NEURAL)
    f.text(663,213,'||',14,anchor='middle',color=NEURAL,bold=True)
    f.bits(523,255,'1010',22)
    f.text(523,287,'Availability m',11,color=MUTED)
    f.poly([(610,266),(663,266),(663,219)],color=MUTED,dash=[3,2])
    f.arrow(674,208,695,208,color=NEURAL)
    f.module(698,166,171,78,'Diagnosis head',['[h; m]: 68 → 64 → 3','GELU + dropout (0.1)','Logits z → softmax'],accent=NEURAL)
    f.arrow(784,244,784,252,color=NEURAL)
    for j,(c,col) in enumerate(zip(['CN','MCI','Dementia'],CLASSCOL)):
        xx=704+j*55
        f.rect(xx,257,51,24,fill=light(col,.88),stroke=col,lw=.7,r=2)
        f.text(xx+25,273,c,10.8,anchor='middle',color=col,bold=True)
    f.text(22,312,'Encoder: Linear(dk, 64) → GELU → LayerNorm → Dropout(0.1) → Linear(64, 64) → GELU.',12,color=MUTED)
    f.save('roi_architecture')

def weighting():
    f=Figure(900,350)
    f.panel(10,10,540,220,'a','Observed diagnosis × availability imbalance',COLORS[0])
    f.panel(565,10,325,220,'b','Training-only modality removal',TRAIN)
    counts={};supported={}
    with COUNTS.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key=(int(row['effective_pattern']),int(row['y']))
            counts[key]=int(row['episodes']);supported[key]=row['supported']=='True'
    assert sum(counts.values())==9566 and sum(supported.values())==34
    maximum=max(counts.values()); x0=89; y0=76; cw=29; ch=28
    f.text(26,60,'9,566 visits · 2,260 participants',12.5,color=MUTED)
    for c,name in enumerate(['CN','MCI','Dementia']):
        f.text(22,y0+c*ch+18,name,12.5,bold=True)
        for p in range(1,16):
            n=counts.get((p,c),0)
            intensity=math.log1p(n)/math.log1p(maximum)
            fill=light(COLORS[0],1-.88*intensity) if n else HexColor('#EDF0F4')
            f.rect(x0+(p-1)*cw,y0+c*ch,cw-1.5,ch-1.5,fill=fill,stroke=white,lw=.5)
            if not supported.get((p,c),False):
                f.circle(x0+(p-1)*cw+cw/2,y0+c*ch+ch/2,2.1,fill=INK,stroke=INK,lw=.3)
    for p in range(1,16):f.text(x0+(p-.5)*cw,y0+3*ch+17,str(p),11.5,anchor='middle',color=MUTED)
    f.text(x0+7.5*cw,193,'Availability pattern ID (MRI, FDG, amyloid, tau)',11.7,anchor='middle',color=MUTED)
    f.circle(24,213,2.1,fill=INK,stroke=INK,lw=.3)
    f.text(32,217,'<20 participants: excluded from SCR',10.7,color=MUTED)
    for j in range(50):
        f.rect(340+j*3.2,205,3.3,8,fill=light(COLORS[0],1-.88*j/49),stroke=None)
    f.text(334,224,'0',10.5,color=MUTED)
    f.text(505,224,str(maximum),10.5,color=MUTED,anchor='end')
    f.text(420,224,'Visits (log scale)',10.5,color=MUTED,anchor='middle')
    f.text(582,60,'Illustration: 100 MRI + FDG examples',12,color=MUTED)
    f.bits(681,71,'1100',23)
    f.text(728,106,'One-modality removal: p = 0.5',11,anchor='middle',color=TRAIN)
    for xx,bits,n,prob in [(581,'1100','50','0.50'),(684,'1000','25','0.25'),(787,'0100','25','0.25')]:
        f.arrow(727,113,xx+42,144,color=TRAIN,lw=.9)
        f.text(xx+42,139,prob,10.5,anchor='middle',color=TRAIN)
        f.bits(xx,151,bits,22)
        f.text(xx+42,194,f'{n} expected',11.5,anchor='middle',bold=True)
    f.text(582,217,'Singletons retain their only modality.',11,color=MUTED)
    f.panel(10,244,880,96,'c','Expected-count joint weighting for the neural branch',NEURAL)
    chain=[('Training-fold counts','n(p, c): observed cells'),('Redistribute expected mass','E = Σ n × T'),('Smooth, clip, normalize','Base: 1 / √(E + 5)'),('Weighted neural loss','mean[w × CE(z, c)]')]
    for i,(title,sub) in enumerate(chain):
        xx=24+i*220
        f.module(xx,284,192,43,title,[sub],accent=TRAIN if i==1 else NEURAL)
        if i<3:f.arrow(xx+192,305,xx+218,305,color=NEURAL)
    f.save('roi_joint_weighting')
    (HERE/'FIGURE_DATA_PROVENANCE.json').write_text(json.dumps({'aggregate_counts_source':str(COUNTS.relative_to(ROOT)),
       'sha256':hashlib.sha256(COUNTS.read_bytes()).hexdigest(),'episodes':sum(counts.values()),'supported_cells':sum(supported.values()),
       'heatmap':'Full development-cohort descriptive counts; fitted weights remain training-fold only.',
       'schematic_feature_tiles':True,'schematic_probability_bars':True,'dropout_example':'Illustrative 100 examples → expected mass 50/25/25; not additional independent subjects.'},indent=2)+'\n')

if __name__=='__main__':
    overview();architecture();weighting()
    print('Created three PDF/SVG vector figures and high-resolution PNG previews.')
