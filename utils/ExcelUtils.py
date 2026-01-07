# -*- coding: utf-8 -*-

import os
import xlwt
import pandas as pd

class XlsReport:
    #    '''Excel报告接口'''
    def __init__(self, file_path):
        self.xls_workbook = None  # 创建的Excel对象
        self.xls_file_path = file_path  # 创建Excel的文件路径
        self.xls_worksheets = {}  # 工作表的行列标记符{worksheet对象:{r:行,c:列}}
        self.xls_sheet_col_length = {}  # 工作表的列宽{worksheet对象:{列1:宽度,列2:宽度...}}

    def xlsOpenWorkbook(self):
        '''创建一个Excel'''
        self.xls_workbook = xlwt.Workbook()

    def xlsAddWorksheet(self, sheet_name='sheet', r=0, c=0):
        '''
        创建一个工作表对象
        @parameter sheet_name:工作簿名称
        @parameter r:工作表的行
        @parameter c:工作表的列
        '''
        obj_worksheet = self.xls_workbook.add_sheet(sheet_name)  # 创建工作表对象
        xls_worksheet = {}  # 工作表的行列标记符
        xls_worksheet['r'] = r  # 行标记符
        xls_worksheet['c'] = c  # 列标记符
        self.xls_worksheets[obj_worksheet] = xls_worksheet
        self.xls_sheet_col_length[obj_worksheet] = {}
        return obj_worksheet

    def xlsCloseWorkbook(self, obj_worksheet):
        '''
        关闭Excel文件对象,保存Excel数据
        @parameter obj_worksheet:工作表对象
        '''
        # 设置工作簿的列宽
        for c in self.xls_sheet_col_length[obj_worksheet]:
            if self.xls_sheet_col_length[obj_worksheet][c] < 10:
                obj_worksheet.col(c).width = 256 * 10
            elif self.xls_sheet_col_length[obj_worksheet][c] > 50:
                obj_worksheet.col(c).width = 256 * 50
            else:
                obj_worksheet.col(c).width = 256 * (self.xls_sheet_col_length[obj_worksheet][c])
        # 关闭工作表对象
        self.xls_workbook.save(self.xls_file_path)

    def addWorksheetTitle(self, obj_worksheet, titles=[], r=0, c=0):
        '''
        添加工作簿的标题
        @parameter obj_worksheet:工作表对象
        @parameter titles:标题
        @parameter r:工作表的行
        @parameter c:工作表的列
        '''
        # 设置标题的样式
        style_title = xlwt.easyxf('pattern:pattern solid,fore_colour lime; font:height 200,bold on; align:horz center;')
        # 写工作簿的标题
        for title in titles:
            obj_worksheet.write(r, c, title, style_title)
            c += 1
        # 工作簿的行标记+1
        r += 1
        self.xls_worksheets[obj_worksheet]['r'] = r

    def appendWorkshetData(self, obj_worksheet, datas=[], r=None, c=None, gold=0):
        '''
        按行追加数据
        @parameter obj_worksheet:工作表对象
        @parameter datas:标题
        @parameter r:工作表的行
        @parameter c:工作表的列
        @parameter gold:0(不变/Pass),-1(变差/Fail),1(变好)
        '''
        # 标准
        stylebox = xlwt.easyxf('font:height 200; borders:left 1,right 1,top 1,bottom 1; align:horiz left,wrap 1')
        # 红体
        stylebox_red = xlwt.easyxf(
            'font:height 200,color-index red; borders:left 1,right 1,top 1,bottom 1; align:horiz left,wrap 1')
        # 蓝体
        stylebox_blue = xlwt.easyxf(
            'font:height 200,color-index blue; borders:left 1,right 1,top 1,bottom 1; align:horiz left,wrap 1')
        # 写工作表的行
        if None == r:
            r = self.xls_worksheets[obj_worksheet]['r']
        else:
            r = r
        # 写工作表的列
        if None == c:
            c = self.xls_worksheets[obj_worksheet]['c']
        else:
            c = c
        # 写工作表的数据
        for data in datas:
            if str != type(data):
                data = str(data)
            if 0 == gold:
                obj_worksheet.write(r, c, data, stylebox)
            elif -1 == gold:
                obj_worksheet.write(r, c, data, stylebox_red)
            elif 1 == gold:
                obj_worksheet.write(r, c, data, stylebox_blue)
            else:
                print('Info:gold was illegal.')
                obj_worksheet.write(r, c, data, stylebox)
            # 工作表每列的最大字符长度
            if c not in self.xls_sheet_col_length[obj_worksheet]:
                self.xls_sheet_col_length[obj_worksheet][c] = len(data)
            else:
                if self.xls_sheet_col_length[obj_worksheet][c] < len(data):
                    self.xls_sheet_col_length[obj_worksheet][c] = len(data)
            c += 1
        # 工作表的行标记+1
        r += 1
        self.xls_worksheets[obj_worksheet]['r'] = r

def generateExcel():
    # 文件的后缀名为xls
    file_path = os.path.join(os.getcwd(), 'UEA30_results.xls')
    xls = XlsReport(file_path)
    # 创建Excel对象
    xls.xlsOpenWorkbook()
    sheet = xls.xlsAddWorksheet('sheet')
    # Excel的标题
    xls.addWorksheetTitle(sheet,['statement', 'test acc'])
    for noise_type in ['sym', 'asym', 'ins']:
        data = pd.read_csv('./data/' + noise_type + '_total.csv')
        acc = (data['state']).astype(str)
        loss = data['test acc']
        xls.appendWorkshetData(sheet, [noise_type, 'v1', '3-3', '4-3', '4-4', '5-3', '5-4', '5-5', '6-3', '6-4', '6-5', '6-6', '7-3', '7-4', '7-5', '7-6', '7-7', '8-3', '8-4', '8-5', '8-6', '8-7', '8-8', '9-3', '9-4', '9-5', '9-6', '9-7', '9-8', '9-9'], gold=0)
        xls.appendWorkshetData(sheet, ['test acc', loss[0], loss[1], loss[2], loss[3], loss[4], loss[5], loss[6], loss[7], loss[8], loss[9], loss[10], loss[11], loss[12], loss[13], loss[14], loss[15], loss[16], loss[17], loss[18], loss[19], loss[20], loss[21], loss[22], loss[23], loss[24], loss[25], loss[26], loss[27], loss[28]], gold=0)
    xls.xlsCloseWorkbook(sheet)


# self test
if __name__ == '__main__':
    generateExcel()
    # # 文件的后缀名为xls
    # file_path = os.path.join(os.getcwd(), 'test.xls')
    # xls = XlsReport(file_path)
    # # 创建Excel对象
    # xls.xlsOpenWorkbook()
    # # 添加工作对象
    # sheet_names = ['sheet1', 'sheet2', 'sheet3']
    # for sheet_name in sheet_names:
    #     sheet = xls.xlsAddWorksheet(sheet_name)
    #     # Excel的标题
    #     xls.addWorksheetTitle(sheet, ['测试用例编号', '测试用例名称', '测试结果', '备注'])
    #     # Excel的数据
    #     xls.appendWorkshetData(sheet, [1001, 'test1', 'Pass', ''], gold=0)
    #     xls.appendWorkshetData(sheet, [1002, 'test2', 'Fail', '失败'], gold=-1)
    #     xls.appendWorkshetData(sheet, [1003, 'test3', 'Pass', '调优'], gold=1)
    #     xls.xlsCloseWorkbook(sheet)