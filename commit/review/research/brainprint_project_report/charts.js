(function () {
  'use strict';

  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg = style.getPropertyValue('--bg').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var accentSoft = style.getPropertyValue('--accent-soft').trim();
  var danger = style.getPropertyValue('--danger').trim();
  var fontFamily = getComputedStyle(document.body).fontFamily;
  var numberFormat = new Intl.NumberFormat('zh-CN');

  function tooltipBase() {
    return {
      appendToBody: true,
      backgroundColor: bg2,
      borderColor: rule,
      borderWidth: 1,
      textStyle: { color: ink, fontFamily: fontFamily }
    };
  }

  function axisBase() {
    return {
      axisLine: { lineStyle: { color: rule } },
      axisTick: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontFamily: fontFamily },
      nameTextStyle: { color: muted, fontFamily: fontFamily },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    };
  }

  function mergeAxis(extra) {
    return Object.assign(axisBase(), extra || {});
  }

  function makeChart(id, option) {
    var chart = echarts.init(document.getElementById(id), null, { renderer: 'svg' });
    option.animation = false;
    option.textStyle = { fontFamily: fontFamily, color: ink };
    chart.setOption(option);
    window.addEventListener('resize', function () { chart.resize(); });
    return chart;
  }

  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'neutral',
      securityLevel: 'loose',
      flowchart: { htmlLabels: true, curve: 'basis', useMaxWidth: false },
      themeVariables: {
        fontFamily: fontFamily,
        primaryColor: accentSoft,
        primaryTextColor: ink,
        primaryBorderColor: accent,
        lineColor: accent2,
        secondaryColor: bg2,
        tertiaryColor: bg
      }
    });
  }

  var taskTooltip = tooltipBase();
  taskTooltip.trigger = 'axis';
  taskTooltip.axisPointer = { type: 'shadow' };
  makeChart('chart-task-distribution', {
    color: [accent, accent2],
    tooltip: taskTooltip,
    legend: { top: 8, textStyle: { color: muted, fontFamily: fontFamily } },
    grid: { left: 76, right: 28, top: 66, bottom: 62, containLabel: false },
    xAxis: mergeAxis({
      type: 'category',
      data: ['EC', 'EO', 'Imagery', 'SSVEP'],
      name: '任务',
      nameLocation: 'middle',
      nameGap: 42,
      splitLine: { show: false }
    }),
    yAxis: mergeAxis({
      type: 'value',
      name: 'Trial 数',
      min: 0,
      max: 12000,
      axisLine: { show: false },
      axisTick: { show: false }
    }),
    series: [
      {
        name: 'Train', type: 'bar', data: [1072, 1070, 5504, 3658],
        barMaxWidth: 48,
        label: { show: true, position: 'top', color: ink, formatter: function (p) { return numberFormat.format(p.value); } }
      },
      {
        name: 'Test', type: 'bar', data: [1972, 2312, 11207, 6760],
        barMaxWidth: 48,
        label: { show: true, position: 'top', color: ink, formatter: function (p) { return numberFormat.format(p.value); } }
      }
    ]
  });

  var bandTooltip = tooltipBase();
  bandTooltip.trigger = 'axis';
  bandTooltip.axisPointer = { type: 'shadow' };
  bandTooltip.valueFormatter = function (value) { return value + ' 错'; };
  makeChart('chart-band-ablation', {
    color: [accent, accent2],
    tooltip: bandTooltip,
    legend: { top: 8, textStyle: { color: muted, fontFamily: fontFamily } },
    grid: { left: 92, right: 64, top: 64, bottom: 58 },
    xAxis: mergeAxis({
      type: 'value', name: '错误数（越少越好）', nameLocation: 'middle', nameGap: 40,
      min: 0, max: 120, axisLine: { show: false }, axisTick: { show: false }
    }),
    yAxis: mergeAxis({
      type: 'category', inverse: true,
      data: ['1–13 Hz', '1–30 Hz', '1–45 Hz', '1–70 Hz', '1–90 Hz（完整）'],
      splitLine: { show: false }
    }),
    series: [
      {
        name: '5折块外推', type: 'bar', data: [107, 9, 6, 8, 9], barMaxWidth: 26,
        label: { show: true, position: 'right', color: ink, formatter: '{c} 错' }
      },
      {
        name: '后20%时间前推', type: 'bar', data: [54, 10, 5, 5, 4], barMaxWidth: 26,
        label: { show: true, position: 'right', color: ink, formatter: '{c} 错' }
      }
    ]
  });

  var evolutionTooltip = tooltipBase();
  evolutionTooltip.trigger = 'axis';
  evolutionTooltip.formatter = function (params) {
    var lines = ['<strong>' + params[0].axisValue + '</strong>'];
    params.forEach(function (p) {
      var value = p.value === null || typeof p.value === 'undefined' ? 'N/A（未运行 / 不适用）' : p.value + ' 错';
      lines.push(p.marker + p.seriesName + '：' + value);
    });
    return lines.join('<br>');
  };
  makeChart('chart-error-evolution', {
    color: [accent, accent2, danger, muted],
    tooltip: evolutionTooltip,
    legend: { top: 8, textStyle: { color: muted, fontFamily: fontFamily } },
    grid: { left: 62, right: 34, top: 78, bottom: 62 },
    xAxis: mergeAxis({
      type: 'category', boundaryGap: true,
      data: ['第一轮', '第二轮', '第三轮', '第四轮'],
      name: '实验轮次', nameLocation: 'middle', nameGap: 42,
      splitLine: { show: false }
    }),
    yAxis: mergeAxis({
      type: 'value', min: 0, max: 10, interval: 2,
      name: '错误数', axisLine: { show: false }, axisTick: { show: false }
    }),
    series: [
      {
        name: '块外推', type: 'line', data: [9, 2, 0, 0], connectNulls: false,
        symbol: 'circle', symbolSize: 10, lineStyle: { width: 3 },
        label: { show: true, position: 'top', color: ink, formatter: '{c}' }
      },
      {
        name: '时间前推', type: 'line', data: [4, 1, 0, 0], connectNulls: false,
        symbol: 'diamond', symbolSize: 11, lineStyle: { width: 3 },
        label: { show: true, position: 'top', color: ink, formatter: '{c}' }
      },
      {
        name: '反向留段', type: 'line', data: [null, null, 2, 0], connectNulls: false,
        symbol: 'triangle', symbolSize: 11, lineStyle: { width: 3 },
        label: { show: true, position: 'top', color: ink, formatter: '{c}' }
      },
      {
        name: '中间留段', type: 'line', data: [null, null, null, 0], connectNulls: false,
        symbol: 'rect', symbolSize: 10, lineStyle: { width: 3 },
        label: { show: true, position: 'bottom', color: ink, formatter: '{c}' }
      }
    ]
  });

  var protocolNames = ['中间40%–60%留段', '首20%反向留段', '后20%时间前推', '5折块外推'];
  var protocolCorrect = [2257, 2245, 2308, 11304];
  var protocolTooltip = tooltipBase();
  protocolTooltip.trigger = 'axis';
  protocolTooltip.axisPointer = { type: 'shadow' };
  protocolTooltip.formatter = function (params) {
    var index = params[0].dataIndex;
    return '<strong>' + protocolNames[index] + '</strong><br>验证样本：' + numberFormat.format(protocolCorrect[index]) + '<br>正确：' + numberFormat.format(protocolCorrect[index]) + '<br>错误：0<br>折外准确率：100%（仅训练 Session 内）';
  };
  makeChart('chart-protocol-validation', {
    color: [accent, danger],
    tooltip: protocolTooltip,
    legend: { top: 8, data: ['折外正确', '错误'], textStyle: { color: muted, fontFamily: fontFamily } },
    grid: { left: 152, right: 76, top: 62, bottom: 58 },
    xAxis: mergeAxis({
      type: 'value', min: 0, max: 12000, name: 'Trial 数', nameLocation: 'middle', nameGap: 40,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: muted, fontFamily: fontFamily, formatter: function (v) { return numberFormat.format(v); } }
    }),
    yAxis: mergeAxis({ type: 'category', data: protocolNames, splitLine: { show: false } }),
    series: [
      {
        name: '折外正确', type: 'bar', stack: 'total', data: protocolCorrect, barMaxWidth: 34,
        label: { show: true, position: 'right', color: ink, formatter: function (p) { return numberFormat.format(p.value) + ' 正确'; } }
      },
      {
        name: '错误', type: 'bar', stack: 'total', data: [0, 0, 0, 0], barMaxWidth: 34,
        label: { show: true, position: 'insideLeft', color: danger, formatter: '0 错' }
      }
    ]
  });

  var sessionTooltip = tooltipBase();
  sessionTooltip.trigger = 'item';
  sessionTooltip.formatter = function (p) {
    return '<strong>' + p.seriesName + '</strong><br>' + p.marker + p.name + '：' + numberFormat.format(p.value) + ' 条（' + p.percent + '%）';
  };
  makeChart('chart-session-fusion', {
    color: [accent, accent2, muted],
    tooltip: sessionTooltip,
    title: [
      { text: '测试 Session 构成', left: '27%', top: '4%', textAlign: 'center', textStyle: { color: ink, fontFamily: fontFamily, fontSize: 15 } },
      { text: 'session2 图融合覆盖', left: '73%', top: '4%', textAlign: 'center', textStyle: { color: ink, fontFamily: fontFamily, fontSize: 15 } }
    ],
    series: [
      {
        name: 'Session 构成', type: 'pie', radius: ['34%', '56%'], center: ['27%', '57%'],
        color: [accent2, accent], avoidLabelOverlap: true,
        itemStyle: { borderColor: bg2, borderWidth: 3 },
        label: { color: ink, fontFamily: fontFamily, formatter: '{b}\n{c} 条（{d}%）' },
        labelLine: { lineStyle: { color: rule } },
        data: [
          { name: 'session1', value: 3033 },
          { name: 'session2', value: 19218 }
        ]
      },
      {
        name: '图融合覆盖', type: 'pie', radius: ['34%', '56%'], center: ['73%', '57%'],
        color: [accent, muted], avoidLabelOverlap: true,
        itemStyle: { borderColor: bg2, borderWidth: 3 },
        label: { color: ink, fontFamily: fontFamily, formatter: '{b}\n{c} 条（{d}%）' },
        labelLine: { lineStyle: { color: rule } },
        data: [
          { name: '高相似覆盖', value: 10205 },
          { name: '未覆盖', value: 9013 }
        ]
      }
    ]
  });

  var labelCounts = [3033, 1083, 400, 1163, 762, 1303, 1230, 1035, 574, 762, 818, 1123, 885, 1214, 867, 1138, 1454, 1095, 735, 692, 885];
  var labelTooltip = tooltipBase();
  labelTooltip.trigger = 'axis';
  labelTooltip.axisPointer = { type: 'shadow' };
  labelTooltip.formatter = function (params) {
    var p = params[0];
    var meaning = p.name === '0' ? '未注册预测' : '注册身份预测';
    return '<strong>标签 ' + p.name + '（' + meaning + '）</strong><br>' + numberFormat.format(p.value) + ' 条';
  };
  makeChart('chart-label-distribution', {
    color: [accent],
    tooltip: labelTooltip,
    grid: { left: 66, right: 24, top: 38, bottom: 70 },
    xAxis: mergeAxis({
      type: 'category', data: Array.from({ length: 21 }, function (_, i) { return String(i); }),
      name: '预测标签', nameLocation: 'middle', nameGap: 48,
      axisLabel: { color: muted, fontFamily: fontFamily, interval: 0, rotate: 45 },
      splitLine: { show: false }
    }),
    yAxis: mergeAxis({
      type: 'value', min: 0, max: 3200, name: '预测 Trial 数',
      axisLine: { show: false }, axisTick: { show: false }
    }),
    series: [{
      name: '预测数量', type: 'bar', data: labelCounts, barMaxWidth: 32,
      itemStyle: { color: function (p) { return p.dataIndex === 0 ? accent2 : accent; } },
      label: { show: true, position: 'top', rotate: 60, color: muted, fontSize: 10, formatter: function (p) { return numberFormat.format(p.value); } },
      markLine: {
        silent: true, symbol: 'none',
        lineStyle: { color: rule, type: 'dashed' },
        label: { color: muted, fontFamily: fontFamily, formatter: '注册身份均值 960.9' },
        data: [{ yAxis: 960.9, name: '注册身份均值' }]
      }
    }]
  });

  var agreementTooltip = tooltipBase();
  agreementTooltip.trigger = 'axis';
  agreementTooltip.axisPointer = { type: 'shadow' };
  agreementTooltip.formatter = function (params) {
    var p = params[0];
    return '<strong>' + p.name + '</strong><br>与直接预测一致率：' + p.value.toFixed(2) + '%<br>仅稳定性，不是准确率';
  };
  makeChart('chart-sinkhorn-agreement', {
    color: [accent],
    tooltip: agreementTooltip,
    grid: { left: 66, right: 34, top: 42, bottom: 62 },
    xAxis: mergeAxis({
      type: 'category', data: ['EC（低信心门控）', 'Imagery', 'SSVEP'],
      name: '任务后处理', nameLocation: 'middle', nameGap: 42,
      splitLine: { show: false }
    }),
    yAxis: mergeAxis({
      type: 'value', min: 0, max: 100, name: '标签一致率（%）',
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: { color: muted, fontFamily: fontFamily, formatter: '{value}%' }
    }),
    series: [{
      name: '一致率', type: 'bar', data: [88.03571428571428, 80.25235945835043, 85.64742589703588], barMaxWidth: 74,
      label: { show: true, position: 'top', color: ink, formatter: function (p) { return p.value.toFixed(2) + '%'; } }
    }]
  });
})();
