//! zenode node definitions for blend and composite operations.
//!
//! Defines [`BlendModeEnum`] (all Porter-Duff and artistic blend modes)
//! and [`Composite`] (overlay one image onto another with blend mode,
//! opacity, and position).

extern crate alloc;
use alloc::string::String;

use zennode::*;

/// Porter-Duff and artistic blend modes.
///
/// All 32 modes from `zenblend::BlendMode`, expressed as a zenode enum
/// for schema introspection and querystring parsing.
#[derive(NodeEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendModeEnum {
    /// Output = 0. Clears both color and alpha.
    Clear,
    /// Output = Src. Background is completely replaced.
    Src,
    /// Output = Dst. Foreground is completely ignored.
    Dst,
    /// Output = Src + Dst * (1 - Src.a). Standard alpha compositing.
    SrcOver,
    /// Output = Dst + Src * (1 - Dst.a).
    DstOver,
    /// Output = Src * Dst.a.
    SrcIn,
    /// Output = Dst * Src.a.
    DstIn,
    /// Output = Src * (1 - Dst.a).
    SrcOut,
    /// Output = Dst * (1 - Src.a).
    DstOut,
    /// Output = Src * Dst.a + Dst * (1 - Src.a).
    SrcAtop,
    /// Output = Dst * Src.a + Src * (1 - Dst.a).
    DstAtop,
    /// Output = Src * (1 - Dst.a) + Dst * (1 - Src.a).
    Xor,

    // Artistic (separable)
    /// Multiply: Src * Dst. Darkens.
    Multiply,
    /// Screen: Src + Dst - Src * Dst. Lightens.
    Screen,
    /// Overlay: Multiply if Dst < 0.5, Screen otherwise.
    Overlay,
    /// Darken: min(Src, Dst).
    Darken,
    /// Lighten: max(Src, Dst).
    Lighten,
    /// Hard light: Multiply if Src < 0.5, Screen otherwise.
    HardLight,
    /// Soft light (W3C formula).
    SoftLight,
    /// Color dodge: Dst / (1 - Src).
    ColorDodge,
    /// Color burn: 1 - (1 - Dst) / Src.
    ColorBurn,
    /// Difference: |Src - Dst|.
    Difference,
    /// Exclusion: Src + Dst - 2 * Src * Dst.
    Exclusion,

    // Additional separable modes
    /// Linear burn: max(0, Src + Dst - 1). Additive darken.
    LinearBurn,
    /// Linear dodge: min(1, Src + Dst). Additive lighten.
    LinearDodge,
    /// Vivid light: ColorBurn(2*Src) if Src < 0.5, ColorDodge(2*Src - 1) otherwise.
    VividLight,
    /// Linear light: LinearBurn(2*Src) if Src < 0.5, LinearDodge(2*Src - 1) otherwise.
    LinearLight,
    /// Pin light: Darken(2*Src) if Src < 0.5, Lighten(2*Src - 1) otherwise.
    PinLight,
    /// Hard mix: 0 or 1 per channel (threshold via VividLight).
    HardMix,
    /// Divide: min(1, Dst / Src). Flat-field correction.
    Divide,
    /// Subtract: max(0, Dst - Src).
    Subtract,
    /// Plus (SVG/CSS): clamp(S + D, 0, 1) on premultiplied values.
    Plus,
}

/// Composite one image onto another with blend mode, opacity, and offset.
///
/// Places the overlay image at (x, y) on the base image, blending with
/// the selected mode and opacity. This is the primary compositing node
/// for layered image pipelines.
///
/// JSON: `{ "blend_mode": "multiply", "opacity": 0.8, "x": 10, "y": 20 }`
/// RIAPI: `?blend_mode=src_over&opacity=0.5`
#[derive(Node, Clone, Debug)]
#[node(id = "zenblend.composite", group = Composite, role = Composite)]
#[node(tags("blend", "composite", "overlay"))]
pub struct Composite {
    /// Blend mode for combining overlay with base.
    ///
    /// Any Porter-Duff or artistic mode from `BlendModeEnum`.
    /// See [`BlendModeEnum`] for the full list of 31 modes.
    #[param(default = "src_over")]
    #[param(section = "Main", label = "Blend Mode")]
    #[kv("blend_mode")]
    pub blend_mode: String,

    /// Overlay opacity (0 = fully transparent, 1 = fully opaque).
    #[param(range(0.0..=1.0), default = 1.0, identity = 1.0, step = 0.01)]
    #[param(section = "Main", label = "Opacity")]
    #[kv("opacity")]
    pub opacity: f32,

    /// Horizontal offset of the overlay in pixels.
    #[param(range(i32::MIN..=i32::MAX), default = 0, identity = 0, step = 1)]
    #[param(unit = "px", section = "Position", label = "X Offset")]
    #[kv("x")]
    pub x: i32,

    /// Vertical offset of the overlay in pixels.
    #[param(range(i32::MIN..=i32::MAX), default = 0, identity = 0, step = 1)]
    #[param(unit = "px", section = "Position", label = "Y Offset")]
    #[kv("y")]
    pub y: i32,
}

impl Default for Composite {
    fn default() -> Self {
        Self {
            blend_mode: String::from("src_over"),
            opacity: 1.0,
            x: 0,
            y: 0,
        }
    }
}

/// Register all zenblend nodes with a registry.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&COMPOSITE_NODE);
}

/// All zenblend zenode definitions.
pub static ALL: &[&dyn NodeDef] = &[&COMPOSITE_NODE];

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    // ─── BlendModeEnum tests ───

    #[test]
    fn blend_mode_variants() {
        let variants = BlendModeEnum::zenode_variants();
        assert_eq!(variants.len(), 32);
        assert_eq!(variants[0].name, "clear");
        assert_eq!(variants[3].name, "src_over");
        assert_eq!(variants[12].name, "multiply");
    }

    #[test]
    fn blend_mode_from_str() {
        assert_eq!(
            BlendModeEnum::from_str("src_over").unwrap(),
            BlendModeEnum::SrcOver
        );
        assert_eq!(
            BlendModeEnum::from_str("multiply").unwrap(),
            BlendModeEnum::Multiply
        );
        assert_eq!(
            BlendModeEnum::from_str("hard_light").unwrap(),
            BlendModeEnum::HardLight
        );
        assert!(BlendModeEnum::from_str("nonexistent").is_err());
    }

    #[test]
    fn blend_mode_display() {
        assert_eq!(BlendModeEnum::SrcOver.to_string(), "src_over");
        assert_eq!(BlendModeEnum::ColorDodge.to_string(), "color_dodge");
        assert_eq!(BlendModeEnum::Plus.to_string(), "plus");
    }

    // ─── Composite node tests ───

    #[test]
    fn composite_schema() {
        let schema = COMPOSITE_NODE.schema();
        assert_eq!(schema.id, "zenblend.composite");
        assert_eq!(schema.group, NodeGroup::Composite);
        assert_eq!(schema.role, NodeRole::Composite);
        assert!(schema.tags.contains(&"blend"));
        assert!(schema.tags.contains(&"composite"));
        assert!(schema.tags.contains(&"overlay"));
    }

    #[test]
    fn composite_defaults() {
        let node = COMPOSITE_NODE.create_default().unwrap();
        assert_eq!(
            node.get_param("blend_mode"),
            Some(ParamValue::Str(String::from("src_over")))
        );
        assert_eq!(node.get_param("opacity"), Some(ParamValue::F32(1.0)));
        assert_eq!(node.get_param("x"), Some(ParamValue::I32(0)));
        assert_eq!(node.get_param("y"), Some(ParamValue::I32(0)));
    }

    #[test]
    fn composite_from_kv() {
        let mut kv = KvPairs::from_querystring("blend_mode=multiply&opacity=0.5&x=10&y=20");
        let node = COMPOSITE_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(
            node.get_param("blend_mode"),
            Some(ParamValue::Str("multiply".into()))
        );
        assert_eq!(node.get_param("opacity"), Some(ParamValue::F32(0.5)));
        assert_eq!(node.get_param("x"), Some(ParamValue::I32(10)));
        assert_eq!(node.get_param("y"), Some(ParamValue::I32(20)));
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn composite_from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("quality=85&width=800");
        let result = COMPOSITE_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn composite_round_trip() {
        let c = Composite {
            blend_mode: String::from("screen"),
            opacity: 0.75,
            x: -50,
            y: 100,
        };
        let params = c.to_params();
        let node = COMPOSITE_NODE.create(&params).unwrap();
        assert_eq!(
            node.get_param("blend_mode"),
            Some(ParamValue::Str("screen".into()))
        );
        assert_eq!(node.get_param("opacity"), Some(ParamValue::F32(0.75)));
        assert_eq!(node.get_param("x"), Some(ParamValue::I32(-50)));
        assert_eq!(node.get_param("y"), Some(ParamValue::I32(100)));
    }

    #[test]
    fn composite_downcast() {
        let node = COMPOSITE_NODE.create_default().unwrap();
        let c = node.as_any().downcast_ref::<Composite>().unwrap();
        assert_eq!(c.blend_mode, "src_over");
        assert_eq!(c.opacity, 1.0);
        assert_eq!(c.x, 0);
        assert_eq!(c.y, 0);
    }

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenblend.composite").is_some());

        let result = registry.from_querystring("blend_mode=overlay&opacity=0.8");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenblend.composite");
    }
}
