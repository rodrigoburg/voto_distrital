select 
	num_zona,
	num_secao,
	partido,
	sum(votos)
from (
	select 
		num_zona,
		num_secao, 
		LEFT(num_votavel,2) as partido, 
		votos 
	from 
		votacao_secao_sp 
	where 
		cod_mun = '71072' and cod_cargo = '13') as sub
group by partido, num_secao, num_zona