"""
结构化医疗信息提取器

从医疗文本中提取:
- 疾病/诊断
- 症状/表现
- 治疗方案
- 药物信息
- 禁忌症
- 检查/检验
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


class EntityType(str, Enum):
    """医疗实体类型"""
    DISEASE = "disease"           # 疾病
    SYMPTOM = "symptom"           # 症状
    TREATMENT = "treatment"       # 治疗方案
    MEDICATION = "medication"     # 药物
    DOSAGE = "dosage"            # 剂量
    CONTRAINDICATION = "contraindication"  # 禁忌症
    EXAMINATION = "examination"   # 检查
    LAB_TEST = "lab_test"        # 化验
    ANATOMY = "anatomy"          # 解剖部位
    PROCEDURE = "procedure"      # 操作/手术


@dataclass
class MedicalEntity:
    """医疗实体"""
    text: str                    # 原文
    entity_type: EntityType      # 实体类型
    normalized: Optional[str] = None  # 标准化形式
    confidence: float = 1.0      # 置信度
    start_pos: Optional[int] = None   # 起始位置
    end_pos: Optional[int] = None     # 结束位置
    attributes: Dict[str, Any] = field(default_factory=dict)  # 属性
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "normalized": self.normalized,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass
class ExtractedInfo:
    """提取的结构化信息"""
    diseases: List[MedicalEntity] = field(default_factory=list)
    symptoms: List[MedicalEntity] = field(default_factory=list)
    treatments: List[MedicalEntity] = field(default_factory=list)
    medications: List[MedicalEntity] = field(default_factory=list)
    contraindications: List[MedicalEntity] = field(default_factory=list)
    examinations: List[MedicalEntity] = field(default_factory=list)
    lab_tests: List[MedicalEntity] = field(default_factory=list)
    procedures: List[MedicalEntity] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "diseases": [e.to_dict() for e in self.diseases],
            "symptoms": [e.to_dict() for e in self.symptoms],
            "treatments": [e.to_dict() for e in self.treatments],
            "medications": [e.to_dict() for e in self.medications],
            "contraindications": [e.to_dict() for e in self.contraindications],
            "examinations": [e.to_dict() for e in self.examinations],
            "lab_tests": [e.to_dict() for e in self.lab_tests],
            "procedures": [e.to_dict() for e in self.procedures],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @property
    def summary(self) -> Dict[str, int]:
        """统计摘要"""
        return {
            "diseases": len(self.diseases),
            "symptoms": len(self.symptoms),
            "treatments": len(self.treatments),
            "medications": len(self.medications),
            "contraindications": len(self.contraindications),
            "examinations": len(self.examinations),
            "lab_tests": len(self.lab_tests),
            "procedures": len(self.procedures),
        }
    
    @property
    def total_entities(self) -> int:
        """实体总数"""
        return sum(self.summary.values())
    
    def get_all_entities(self) -> List[MedicalEntity]:
        """获取所有实体"""
        return (
            self.diseases + self.symptoms + self.treatments +
            self.medications + self.contraindications +
            self.examinations + self.lab_tests + self.procedures
        )


class MedicalExtractor:
    """
    医疗信息提取器
    
    支持两种模式:
    1. 基于规则的快速提取（默认）
    2. 基于 LLM 的深度提取（需要配置 LLM）
    
    示例:
        ```python
        extractor = MedicalExtractor()
        
        # 基于规则提取
        info = extractor.extract("患者诊断为2型糖尿病，建议服用二甲双胍...")
        print(info.to_json())
        
        # 基于 LLM 提取
        info = await extractor.extract_with_llm(text, llm_client)
        ```
    """
    
    def __init__(self):
        # 常见疾病词典
        self._disease_patterns = self._build_disease_patterns()
        # 症状词典
        self._symptom_patterns = self._build_symptom_patterns()
        # 药物词典
        self._medication_patterns = self._build_medication_patterns()
        # 检查词典
        self._examination_patterns = self._build_examination_patterns()
        # 治疗词典
        self._treatment_patterns = self._build_treatment_patterns()
        # 禁忌词典
        self._contraindication_patterns = self._build_contraindication_patterns()
    
    def _build_disease_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建疾病模式"""
        diseases = [
            # 心血管
            (r"高血压", "高血压"),
            (r"冠心病|冠状动脉粥样硬化性心脏病", "冠心病"),
            (r"心肌梗[死塞]|心梗", "心肌梗死"),
            (r"心[房室]颤动|房颤", "心房颤动"),
            (r"心力衰竭|心衰", "心力衰竭"),
            (r"心绞痛", "心绞痛"),
            # 内分泌
            (r"[12I]型糖尿病|糖尿病", "糖尿病"),
            (r"甲[状腺]*亢[进]?", "甲状腺功能亢进"),
            (r"甲[状腺]*减[退]?", "甲状腺功能减退"),
            (r"甲状腺结节", "甲状腺结节"),
            # 呼吸系统
            (r"肺炎", "肺炎"),
            (r"支气管炎", "支气管炎"),
            (r"哮喘", "支气管哮喘"),
            (r"慢性阻塞性肺[病疾]|COPD|慢阻肺", "慢性阻塞性肺疾病"),
            (r"肺结核", "肺结核"),
            # 消化系统
            (r"胃炎", "胃炎"),
            (r"胃溃疡", "胃溃疡"),
            (r"肝炎", "肝炎"),
            (r"肝硬化", "肝硬化"),
            (r"胆囊炎", "胆囊炎"),
            (r"胆结石|胆石症", "胆石症"),
            (r"幽门螺[杆旋]菌感染|Hp感染", "幽门螺杆菌感染"),
            # 神经系统
            (r"脑卒中|中风|脑梗[死塞]?", "脑卒中"),
            (r"帕金森[病症]?", "帕金森病"),
            (r"阿尔茨海默[病症]?|老年痴呆", "阿尔茨海默病"),
            (r"癫痫", "癫痫"),
            (r"偏头痛", "偏头痛"),
            # 骨骼肌肉
            (r"骨质疏松[症]?", "骨质疏松症"),
            (r"类风湿[性]?关节炎", "类风湿关节炎"),
            (r"骨关节炎", "骨关节炎"),
            (r"腰椎间盘突出", "腰椎间盘突出"),
            # 肾脏
            (r"慢性肾[脏病功能不全衰竭]|CKD", "慢性肾脏病"),
            (r"肾结石", "肾结石"),
            # 肿瘤
            (r"肺癌", "肺癌"),
            (r"胃癌", "胃癌"),
            (r"肝癌", "肝癌"),
            (r"乳腺癌", "乳腺癌"),
            (r"结直肠癌|大肠癌", "结直肠癌"),
            # 精神
            (r"抑郁[症障碍]?", "抑郁症"),
            (r"焦虑[症障碍]?", "焦虑症"),
            (r"失眠[症]?", "失眠"),
            # 传染病
            (r"新冠|COVID-?19|新型冠状病毒", "新型冠状病毒感染"),
            (r"流感|流行性感冒", "流行性感冒"),
        ]
        return [(re.compile(pattern, re.IGNORECASE), norm) for pattern, norm in diseases]
    
    def _build_symptom_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建症状模式"""
        symptoms = [
            # 全身症状
            (r"发热|发烧|体温升高", "发热"),
            (r"乏力|疲劳|无力", "乏力"),
            (r"体重[下降减轻增加]", "体重变化"),
            (r"水肿|浮肿", "水肿"),
            (r"盗汗", "盗汗"),
            # 疼痛
            (r"头痛|头疼", "头痛"),
            (r"胸痛|胸闷", "胸痛"),
            (r"腹痛|肚子痛", "腹痛"),
            (r"关节痛", "关节痛"),
            (r"腰痛|腰酸", "腰痛"),
            # 呼吸系统
            (r"咳嗽", "咳嗽"),
            (r"咳痰", "咳痰"),
            (r"呼吸困难|气短|气促", "呼吸困难"),
            (r"喘息|气喘", "喘息"),
            # 消化系统
            (r"恶心", "恶心"),
            (r"呕吐", "呕吐"),
            (r"腹泻|拉肚子", "腹泻"),
            (r"便秘", "便秘"),
            (r"食欲[不振下降减退]", "食欲减退"),
            # 心血管
            (r"心悸|心慌", "心悸"),
            (r"眩晕|头晕", "眩晕"),
            # 泌尿
            (r"尿频", "尿频"),
            (r"尿急", "尿急"),
            (r"尿痛", "尿痛"),
            (r"血尿", "血尿"),
            # 神经
            (r"失眠|睡眠障碍", "失眠"),
            (r"记忆力[下降减退]", "记忆力下降"),
            (r"麻木", "麻木"),
            # 皮肤
            (r"皮疹", "皮疹"),
            (r"瘙痒", "瘙痒"),
            (r"黄疸", "黄疸"),
        ]
        return [(re.compile(pattern, re.IGNORECASE), norm) for pattern, norm in symptoms]
    
    def _build_medication_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建药物模式"""
        medications = [
            # 降糖药
            (r"二甲双胍", "二甲双胍"),
            (r"格列[美苯吡奈齐]脲?", "磺脲类降糖药"),
            (r"胰岛素", "胰岛素"),
            (r"阿卡波糖", "阿卡波糖"),
            # 降压药
            (r"氨氯地平", "氨氯地平"),
            (r"硝苯地平", "硝苯地平"),
            (r"缬沙坦", "缬沙坦"),
            (r"氯沙坦", "氯沙坦"),
            (r"依那普利", "依那普利"),
            (r"卡托普利", "卡托普利"),
            (r"美托洛尔", "美托洛尔"),
            (r"比索洛尔", "比索洛尔"),
            # 降脂药
            (r"阿托伐他汀", "阿托伐他汀"),
            (r"瑞舒伐他汀", "瑞舒伐他汀"),
            (r"辛伐他汀", "辛伐他汀"),
            # 抗血小板
            (r"阿司匹林", "阿司匹林"),
            (r"氯吡格雷", "氯吡格雷"),
            # 抗生素
            (r"阿莫西林", "阿莫西林"),
            (r"头孢[类克曲唑]*", "头孢类抗生素"),
            (r"青霉素", "青霉素"),
            (r"阿奇霉素", "阿奇霉素"),
            (r"左氧氟沙星", "左氧氟沙星"),
            (r"甲硝唑", "甲硝唑"),
            (r"克拉霉素", "克拉霉素"),
            # 消化
            (r"奥美拉唑", "奥美拉唑"),
            (r"雷贝拉唑", "雷贝拉唑"),
            (r"法莫替丁", "法莫替丁"),
            # 镇痛
            (r"布洛芬", "布洛芬"),
            (r"对乙酰氨基酚|扑热息痛", "对乙酰氨基酚"),
            # 激素
            (r"泼尼松|强的松", "泼尼松"),
            (r"地塞米松", "地塞米松"),
            # 其他
            (r"维生素[A-Z]|维[A-Z]", "维生素"),
            (r"钙[剂片]", "钙剂"),
        ]
        return [(re.compile(pattern, re.IGNORECASE), norm) for pattern, norm in medications]
    
    def _build_examination_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建检查模式"""
        examinations = [
            # 影像
            (r"CT|CT检查|计算机断层", "CT检查"),
            (r"MRI|核磁共振", "MRI检查"),
            (r"X[光线]?|胸片", "X线检查"),
            (r"B超|超声", "超声检查"),
            (r"心电图|ECG|EKG", "心电图"),
            (r"胃镜", "胃镜检查"),
            (r"肠镜", "肠镜检查"),
            # 化验
            (r"血常规", "血常规"),
            (r"尿常规", "尿常规"),
            (r"肝功[能检查]?", "肝功能"),
            (r"肾功[能检查]?", "肾功能"),
            (r"血糖", "血糖"),
            (r"糖化血红蛋白|HbA1c", "糖化血红蛋白"),
            (r"血脂", "血脂"),
            (r"甲[状腺]?功[能]?", "甲状腺功能"),
            (r"肿瘤标志物", "肿瘤标志物"),
            (r"血压", "血压测量"),
        ]
        return [(re.compile(pattern, re.IGNORECASE), norm) for pattern, norm in examinations]
    
    def _build_treatment_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建治疗模式"""
        treatments = [
            (r"手术[治疗]?", "手术治疗"),
            (r"化疗|化学治疗", "化学治疗"),
            (r"放疗|放射治疗", "放射治疗"),
            (r"免疫治疗", "免疫治疗"),
            (r"靶向治疗", "靶向治疗"),
            (r"透析", "透析治疗"),
            (r"支架[植入置入]?", "支架植入"),
            (r"搭桥", "搭桥手术"),
            (r"康复[治疗训练]?", "康复治疗"),
            (r"物理治疗", "物理治疗"),
            (r"中医治疗", "中医治疗"),
            (r"针灸", "针灸治疗"),
            (r"按摩|推拿", "推拿治疗"),
        ]
        return [(re.compile(pattern, re.IGNORECASE), norm) for pattern, norm in treatments]
    
    def _build_contraindication_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """构建禁忌模式"""
        patterns = [
            (r"禁[忌用]于", "禁忌"),
            (r"不[宜能适]用于", "不宜使用"),
            (r"慎用", "慎用"),
            (r"过敏", "过敏"),
            (r"妊娠期禁用|孕妇禁用", "妊娠期禁用"),
            (r"哺乳期禁用", "哺乳期禁用"),
            (r"肝[功能]?损[害伤]", "肝功能损害"),
            (r"肾[功能]?损[害伤]", "肾功能损害"),
        ]
        return [(re.compile(pattern, re.IGNORECASE), norm) for pattern, norm in patterns]
    
    def _extract_by_patterns(
        self,
        text: str,
        patterns: List[Tuple[re.Pattern, str]],
        entity_type: EntityType,
    ) -> List[MedicalEntity]:
        """基于模式提取实体"""
        entities = []
        seen: Set[str] = set()
        
        for pattern, normalized in patterns:
            for match in pattern.finditer(text):
                matched_text = match.group()
                # 去重
                key = f"{normalized}:{match.start()}"
                if key in seen:
                    continue
                seen.add(key)
                
                entities.append(MedicalEntity(
                    text=matched_text,
                    entity_type=entity_type,
                    normalized=normalized,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                ))
        
        return entities
    
    def extract(self, text: str) -> ExtractedInfo:
        """
        基于规则提取医疗信息
        
        Args:
            text: 输入文本
            
        Returns:
            ExtractedInfo 结构化信息
        """
        info = ExtractedInfo(raw_text=text)
        
        # 提取各类实体
        info.diseases = self._extract_by_patterns(
            text, self._disease_patterns, EntityType.DISEASE
        )
        info.symptoms = self._extract_by_patterns(
            text, self._symptom_patterns, EntityType.SYMPTOM
        )
        info.medications = self._extract_by_patterns(
            text, self._medication_patterns, EntityType.MEDICATION
        )
        info.examinations = self._extract_by_patterns(
            text, self._examination_patterns, EntityType.EXAMINATION
        )
        info.treatments = self._extract_by_patterns(
            text, self._treatment_patterns, EntityType.TREATMENT
        )
        
        # 提取禁忌相关
        contraindication_entities = self._extract_by_patterns(
            text, self._contraindication_patterns, EntityType.CONTRAINDICATION
        )
        info.contraindications = contraindication_entities
        
        # 提取剂量信息
        dosage_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*(mg|g|ml|片|粒|支|次/[日天]|[日天]/次)',
            re.IGNORECASE
        )
        for match in dosage_pattern.finditer(text):
            # 关联到最近的药物
            pass  # 简化处理，完整实现需要更复杂的逻辑
        
        logger.debug(f"提取完成: {info.summary}")
        return info
    
    async def extract_with_llm(
        self,
        text: str,
        llm_client,
        prompt_template: Optional[str] = None,
    ) -> ExtractedInfo:
        """
        基于 LLM 提取医疗信息（更准确但更慢）
        
        Args:
            text: 输入文本
            llm_client: LLM 客户端（需要有 generate 方法）
            prompt_template: 提示词模板
            
        Returns:
            ExtractedInfo 结构化信息
        """
        if prompt_template is None:
            prompt_template = """请从以下医疗文本中提取结构化信息，以 JSON 格式输出：

文本：
{text}

请提取以下类别的信息：
1. diseases - 疾病/诊断
2. symptoms - 症状/表现
3. medications - 药物
4. treatments - 治疗方案
5. examinations - 检查项目
6. contraindications - 禁忌症

输出格式：
{{
  "diseases": ["疾病1", "疾病2"],
  "symptoms": ["症状1", "症状2"],
  "medications": ["药物1", "药物2"],
  "treatments": ["治疗1", "治疗2"],
  "examinations": ["检查1", "检查2"],
  "contraindications": ["禁忌1", "禁忌2"]
}}

只输出 JSON，不要其他内容。"""
        
        prompt = prompt_template.format(text=text[:2000])  # 限制长度
        
        try:
            # 调用 LLM
            if hasattr(llm_client, 'generate_sync'):
                response = llm_client.generate_sync(prompt, [])
            elif hasattr(llm_client, 'generate'):
                response = await llm_client.generate(prompt, [])
            else:
                raise ValueError("LLM client must have generate or generate_sync method")
            
            # 解析 JSON
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                logger.warning("LLM 输出中未找到 JSON")
                return self.extract(text)  # fallback 到规则提取
            
            # 构建结果
            info = ExtractedInfo(raw_text=text)
            
            for disease in data.get("diseases", []):
                info.diseases.append(MedicalEntity(
                    text=disease,
                    entity_type=EntityType.DISEASE,
                    normalized=disease,
                ))
            
            for symptom in data.get("symptoms", []):
                info.symptoms.append(MedicalEntity(
                    text=symptom,
                    entity_type=EntityType.SYMPTOM,
                    normalized=symptom,
                ))
            
            for medication in data.get("medications", []):
                info.medications.append(MedicalEntity(
                    text=medication,
                    entity_type=EntityType.MEDICATION,
                    normalized=medication,
                ))
            
            for treatment in data.get("treatments", []):
                info.treatments.append(MedicalEntity(
                    text=treatment,
                    entity_type=EntityType.TREATMENT,
                    normalized=treatment,
                ))
            
            for examination in data.get("examinations", []):
                info.examinations.append(MedicalEntity(
                    text=examination,
                    entity_type=EntityType.EXAMINATION,
                    normalized=examination,
                ))
            
            for contraindication in data.get("contraindications", []):
                info.contraindications.append(MedicalEntity(
                    text=contraindication,
                    entity_type=EntityType.CONTRAINDICATION,
                    normalized=contraindication,
                ))
            
            return info
            
        except Exception as e:
            logger.error(f"LLM 提取失败: {e}")
            return self.extract(text)  # fallback


# ============== 便捷函数 ==============

def extract_medical_info(text: str) -> ExtractedInfo:
    """
    提取医疗信息的便捷函数
    
    Args:
        text: 输入文本
        
    Returns:
        ExtractedInfo
    """
    extractor = MedicalExtractor()
    return extractor.extract(text)


def extract_to_json(text: str, indent: int = 2) -> str:
    """
    提取医疗信息并返回 JSON
    
    Args:
        text: 输入文本
        indent: JSON 缩进
        
    Returns:
        JSON 字符串
    """
    info = extract_medical_info(text)
    return info.to_json(indent=indent)
